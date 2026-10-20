"""Audio-only native MFA decoding and external TIMIT metrics, with optional K2 FA."""
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import jiwer
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segmentation_metrics import r_value_percent

DATA=Path(__file__).resolve().parent/'mfa_eval_data'

def audio_forward(net,arm,x,lengths):
    """No target sequence or timestamps enter the acoustic forward."""
    if arm!='film_k4':
        enc,out_lengths=net.acoustic.forward_encoder(x,lengths,aux_loss_scale=0.)
        logits=net.acoustic.ctc_output(enc).float()
        forbidden=list(range(1,7)) if arm=='spottc' else [1,2]
        logits=logits.index_fill(-1,torch.tensor(forbidden,device=x.device),-1e4)
        masses=net.weight_head(enc.float()).squeeze(-1).softmax(-1) if arm!='ctc' else None
    else:
        from film_k4_10ms import acoustic_logits
        h,out_lengths=net.frontend(x,lengths);h=h.transpose(0,1);original=h
        mask=torch.arange(h.shape[0],device=x.device)[None]>=lengths[:,None]
        for k in range(1,5):
            z=net.encoder.encoders[0](h,src_key_padding_mask=mask,aux_loss_scale=0.)
            normalized=net.encoder.out_norm(z).transpose(0,1)
            if k<4:
                phone_logits=acoustic_logits(net.phone_head(normalized))
                h=net.transition(z,original,phone_logits.transpose(0,1),k)
        logits=acoustic_logits(net.final_head(normalized),states=True)
        masses=net.weight_head(normalized.float()).squeeze(-1).softmax(-1)
    assert logits.shape[1]==x.shape[1] and torch.equal(out_lengths,lengths)
    return logits,out_lengths,masses

def greedy(path):
    return [v for i,v in enumerate(path) if v and (i==0 or v!=path[i-1])]

def match(ref,pred,tol):
    i=j=m=0
    while i<len(ref) and j<len(pred):
        if abs(ref[i]-pred[j])<=tol+1e-9:m+=1;i+=1;j+=1
        elif ref[i]<pred[j]:i+=1
        else:j+=1
    return m

def boundary_scores(count,reference,predicted,duration):
    ref=sorted({t for _,s,e in reference for t in [s,e] if .02<t<duration-.02})
    pred=sorted({float(t) for t in predicted if .02<t<duration-.02})
    count['reference_boundaries']+=len(ref);count['predicted_boundaries']+=len(pred)
    for ms in [10,20,30]:count['matches_'+str(ms)]+=match(ref,pred,ms/1000)

def summarize(count):
    out=dict(count)
    n=count['reference_phones'];out['PER_percent']=100*count['phone_errors']/n if n else None
    r,p=count['reference_boundaries'],count['predicted_boundaries']
    for ms in [10,20,30]:
        m=count['matches_'+str(ms)]
        out['boundary_precision_'+str(ms)+'ms_percent']=100*m/p if p else 0.
        out['boundary_recall_'+str(ms)+'ms_percent']=100*m/r if r else 0.
        out['boundary_F1_'+str(ms)+'ms_percent']=200*m/(r+p) if r+p else 0.
        if r:
            out['boundary_R_value_'+str(ms)+'ms_percent']=r_value_percent(
                out['boundary_precision_'+str(ms)+'ms_percent'],out['boundary_recall_'+str(ms)+'ms_percent'])
    if count['aligned_tokens']:
        n=count['aligned_tokens'];out.update(start_MAE_ms=1000*count['start_error']/n,end_MAE_ms=1000*count['end_error']/n,mean_token_IoU_percent=100*count['iou']/n)
    return out

def evaluate(net,arm,split='development',rank=0,world=1,forced=False,limit=0,output=None):
    """Shard dev across the existing training ranks; never advance the optimizer."""
    from ifmdd_losses import subphonetic_greedy_complete
    rows=[json.loads(line) for line in (DATA/(split+'.jsonl')).open()]
    if limit:rows=rows[:limit]
    counts=Counter();fa=Counter();caches={};start=time.monotonic()
    backend=None;provenance=None
    if forced:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from ifmdd_k2_alignment import load_backend,align
        backend,provenance=load_backend()
        if output:
            (output/'ifmdd_k2_source_snapshot.py').write_text(provenance.pop('source_snapshot')+'\n')
    prediction_file=(output/(split+'_predictions.jsonl')).open('w') if output else None
    try:
        with torch.no_grad():
            for index,row in enumerate(rows):
                if index%world!=rank:continue
                path=row['feature_file']
                if path not in caches:caches[path]=np.memmap(path,dtype=np.float32,mode='r')
                feature=np.array(caches[path][row['feature_offset']:row['feature_offset']+row['frames']*80]).reshape(-1,80)
                x=torch.from_numpy(feature).unsqueeze(0).cuda();lengths=torch.tensor([len(feature)],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,out_lengths,masses=audio_forward(net,arm,x,lengths)
                scores=logits[0,:len(feature)].float();raw=scores.argmax(-1)
                states=arm in ['spottc','film_k4']
                if states:
                    hyp=subphonetic_greedy_complete(raw.unsqueeze(0),out_lengths,'s3_t2')[0]
                    phone_path=torch.where(raw==0,0,(raw-1)//3+1).cpu().tolist()
                    # A phone marginal posterior allows the same IF-MDD K2 CTC graph across all arms.
                    logp=scores.log_softmax(-1)
                    phone_scores=torch.cat([logp[:,:1],torch.logsumexp(logp[:,1:].reshape(len(feature),104,3),-1)],-1)
                else:hyp=greedy(raw.cpu().tolist());phone_path=raw.cpu().tolist();phone_scores=scores.log_softmax(-1)
                hyp=[v for v in hyp if 3<=v<=102]
                ref=row['tokens'];e=jiwer.process_words(' '.join(map(str,ref)),' '.join(map(str,hyp)))
                counts.update(utterances=1,reference_phones=len(ref),phone_errors=e.substitutions+e.insertions+e.deletions,insertions=e.insertions,deletions=e.deletions,substitutions=e.substitutions)
                edges=[i*.01 for i in range(1,len(phone_path)) if phone_path[i]!=phone_path[i-1]]
                boundary_scores(counts,row['references'],edges,row['duration'])
                result={'id':row['id'],'audio_only':True,'predicted_ids':hyp,'frame_phone_ids':phone_path}
                if forced:
                    try:
                        # Keep annotated silence in the supplied sequence so outer
                        # speech tokens do not absorb the leading/trailing silence.
                        supplied=[p['mfa_id'] for p in row['phone_intervals']]
                        spans,k2path=align(phone_scores,supplied,row['duration'],backend)
                        segments=[(s['token_id'],s['start'],s['end']) for s,p in zip(spans,row['phone_intervals']) if p['score_speech']]
                        assert [p for p,_,_ in segments]==ref
                        boundary_scores(fa,row['references'],[t for _,s,e in segments for t in [s,e]],row['duration'])
                        for (_,rs,re),(_,ps,pe) in zip(row['references'],segments):
                            union=max(re,pe)-min(rs,ps)
                            fa.update(aligned_tokens=1,start_error=abs(rs-ps),end_error=abs(re-pe),iou=max(0,min(re,pe)-max(rs,ps))/union if union else 0.)
                        fa['utterances']+=1
                        result['forced_k2']={'spans_seconds':segments,'supplied_ids':supplied,'frame_ids':k2path,'reference_times_used':False}
                    except ValueError as exc:
                        fa['alignment_failures']+=1;result['forced_k2']={'failure':str(exc)}
                if prediction_file:prediction_file.write(json.dumps(result)+'\n')
                if (index+1)%500==0:print(arm,split,index+1,'/',len(rows),flush=True)
    finally:
        if prediction_file:prediction_file.close()
    if world>1:
        local={'text_free':dict(counts),'forced_k2':dict(fa)};gathered=[None]*world
        dist.all_gather_object(gathered,local)
        counts=Counter();fa=Counter()
        for item in gathered:counts.update(item['text_free']);fa.update(item['forced_k2'])
    result={'text_free':summarize(counts),'elapsed_seconds':time.monotonic()-start,'split':split,'limit':limit,
            'protocol':{'recognition':'Audio only; native CTC collapse or original S3T2 greedy complete; remove non-speech IDs >=103','segmentation':'Phone-level argmax changes including blank, 10ms grid; substate changes inside a phone excluded','reference':'Frozen original TIMIT-to-MFA mapping, closure/release fusion; representative phone PER includes annotation-system mismatch','test_selection':False,'fa':'IF-MDD K2-FSA native CTC phone graph; SPOTTC/FiLM use sum of substate posteriors; previous blank, duration/frames','k2_backend':provenance}}
    import hashlib
    from segmentation_metrics import SOURCE as r_value_source
    result['protocol']['R_value']={'source':str(r_value_source),
        'source_sha256':hashlib.sha256(r_value_source.read_bytes()).hexdigest(),
        'definition':'R-value from recorded P/R; existing boundary extraction and matching unchanged'}
    if forced:
        result['forced_k2']=summarize(fa);result['forced_k2'].pop('PER_percent',None)
        assert not fa['alignment_failures'],dict(fa)
    return result
