"""Libri100 development/test audio-only recognition and native K2 phone FA."""
import bootstrap
from bootstrap import ROOT, HERE
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from ifmdd_k2_alignment import load_backend, align
from segmentation_metrics import r_value_percent, SOURCE as R_SOURCE
from train import model_directory, OBJECTIVES, boundary_matches, phone_per

EVALUATION_SPLITS = ['dev_clean','dev_other','test_clean','test_other','buckeye_dev','buckeye_test']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def edges(spans,duration):
    return sorted({float(t) for _,s,e in spans for t in (s,e) if .02<t<duration-.02})


def new_counts():
    return {'utterances':0,'alignment_failures':0,'bounds':{str(ms):Counter() for ms in [10,20,30]},
        'refs':[],'hyps':[],'errors':Counter()}


def score(counts,reference,predicted,duration):
    rb=edges(reference,duration)
    pb=sorted(set(float(t) for t in predicted if .02<t<duration-.02))
    for ms,c in counts['bounds'].items():
        c['matches']+=boundary_matches(rb,pb,int(ms)/1000)
        c['reference']+=len(rb);c['predicted']+=len(pb)


def summary(counts):
    result={k:counts[k] for k in ['utterances','alignment_failures']}
    result['boundaries']={}
    for ms,c in counts['bounds'].items():
        m,r,p=c['matches'],c['reference'],c['predicted']
        precision=100*m/p if p else 0.;recall=100*m/r if r else 0.
        result['boundaries'][ms]=dict(c,precision_percent=precision,recall_percent=recall,
            f1_percent=200*m/(r+p) if r+p else 0.,r_value_percent=r_value_percent(precision,recall))
    if counts['refs']:result['nosil_PER_percent']=phone_per(counts['refs'],counts['hyps'])
    err=counts['errors'];n=err['tokens']
    if n:result.update(aligned_tokens=n,start_MAE_ms=1000*err['start']/n,end_MAE_ms=1000*err['end']/n,
        boundary_MAE_ms=500*(err['start']+err['end'])/n,start_plus_end_TSE_ms=1000*(err['start']+err['end'])/n,
        start_hit20_percent=100*err['start_hit20']/n,end_hit20_percent=100*err['end_hit20']/n)
    return result


def publish(result,output,smoke):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    options=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
        name='libri100-singlehead-'+result['objective']+'-'+result['artifact_label']+'-K2FA'+('-smoke' if smoke else ''),
        group='libri100_singlehead_average_20261016',job_type='evaluation',
        tags=['libri100','singlehead','phone-reference-FA','smoke' if smoke else 'full'],
        dir=str(output),mode=os.environ.get('WANDB_MODE','online'),config=result['protocol'],
        settings=wandb.Settings(init_timeout=45,disable_code=True))
    identity=output/'wandb_run.json'
    if identity.exists():
        saved=json.loads(identity.read_text());options['id']=saved['id']
        if options['mode']=='online':options['resume']='allow'
    try:run=init_evaluation(**options)
    except wandb.errors.CommError:
        options['mode']='offline';options.pop('resume',None);run=init_evaluation(**options)
    (output/'wandb_run.json').write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode),indent=2)+'\n')
    values={}
    for split,modes in result['splits'].items():
        for mode,metrics in modes.items():
            for key,value in metrics.items():
                if key=='boundaries':
                    for ms,b in value.items():
                        for metric,v in b.items():values[f'benchmark/{split}/{mode}/{metric}_{ms}ms']=v
                elif isinstance(value,(float,int)):values[f'benchmark/{split}/{mode}/{key}']=value
    run.log(values);run.summary.update(values)
    artifact=wandb.Artifact('libri100-singlehead-'+result['objective']+'-'+result['artifact_label']+'-K2FA'+('-smoke' if smoke else ''),type='evaluation')
    for path in output.glob('*.json*'):
        if path.name!='wandb_run.json':artifact.add_file(str(path),name=path.name)
    for name in ['ifmdd_k2_source_snapshot.py','evaluator_snapshot.py']:artifact.add_file(str(output/name),name=name)
    run.log_artifact(artifact);run.finish()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=list(OBJECTIVES),required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=EVALUATION_SPLITS,default=EVALUATION_SPLITS)
    parser.add_argument('--label',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    assert torch.cuda.is_available()
    directory=model_directory(args.objective,False)
    checkpoint=args.checkpoint.resolve()
    output=args.output.resolve()
    assert (checkpoint/'model.safetensors').exists()
    if (output/'EVAL_COMPLETE.json').exists():
        saved=json.loads((output/'EVAL_COMPLETE.json').read_text())
        assert saved['checkpoint']==str(checkpoint) and saved['smoke']==args.smoke
        assert set(saved['utterances'])==set(args.splits)
        print('Evaluation already complete; reuse',output,flush=True);return
    output.mkdir(parents=True,exist_ok=True)
    model,loading=wvForOTTC.from_pretrained(checkpoint,output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
    assert model.config.bins_metric==OBJECTIVES[args.objective]
    model.eval().cuda();model.step=int(checkpoint.name.split('-')[1])
    processor=Wav2Vec2Processor.from_pretrained(directory/'processor')
    vocab=json.loads((directory/'vocab_used.json').read_text());inverse={v:k for k,v in vocab.items()}
    backend,provenance=load_backend()
    (output/'ifmdd_k2_source_snapshot.py').write_text(provenance.pop('source_snapshot')+'\n')
    (output/'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    result=dict(objective=args.objective,artifact_label=args.label,checkpoint=str(checkpoint),checkpoint_loading_info=loading,
        checkpoint_sha256=digest(checkpoint/'model.safetensors'),evaluator_sha256=digest(Path(__file__)),
        protocol=dict(corpus='LibriSpeech train-clean-100; held-out clean/other and zero-shot original-human Buckeye partial19',
            reference_sources={'LibriSpeech':'existing automatic alignment; model identity unverified; not human boundaries', 'Buckeye':'same original human partial19 as Buckeye training experiment'},
            frame_ms=20,phone_mapping={'ao':'aa','zh':'sh'},forced_input='Exact mapped reference speech-phone sequence; no reference times or silence targets',
            inference='Audio-only FP32 native42-class logits; CE versus original upstream JSD versus standard JSD; no retraining during evaluation',
            timestamp_grid='duration/frame_count; blank previous; no offset',k2=provenance,
            boundary_protocol='unique interval endpoints, exclude utterance edges20ms, one-to-one time-only matching at10/20/30ms',
            failure_policy='Keep all utterances; failed FA contributes reference boundaries and zero matches; silence-only reference is recorded as inapplicable FA; PER counts its insertions; timestamp MAE successful exact-target alignments',
            selection='Retained single checkpoint or uniform weight average; dev-only selection; see selection.json',source_evaluator_sha256='1e285d1298424abe9042fcacea71f266b0d94308e8bacd78fda873bf2c1405f3',reference_times_used_by_aligner=False,
            optimizer_updated=False,word_transcript_FA_Bench=False,r_value_source_sha256=digest(R_SOURCE)),splits={})
    begin=time.monotonic();independence_checked=False
    with torch.inference_mode():
        for split in args.splits:
            manifest=HERE/'data'/(split+'.jsonl')
            rows=[json.loads(l) for l in manifest.read_text().splitlines()]
            if args.smoke:rows=rows[:8]
            totals={mode:new_counts() for mode in ['text_free','forced_k2']}
            with (output/(split+'_predictions.jsonl')).open('w') as f:
                for i,row in enumerate(rows):
                    audio,sr=sf.read(row['wav'],dtype='float32');assert sr==16000 and audio.ndim==1
                    batch=processor(audio,sampling_rate=sr,return_tensors='pt',return_attention_mask=True).to('cuda')
                    out=model(**batch)
                    length=int(out.hidden_states[1][0].sum());logits=out.logits[0,:length].float()
                    if not independence_checked:
                        labels=torch.tensor([row['training_target_ids']],dtype=torch.long,device='cuda')
                        check=model(**batch,labels=labels)
                        torch.testing.assert_close(out.logits,check.logits,atol=0,rtol=0)
                        independence_checked=True
                    path=logits.argmax(-1).cpu().numpy();path[path==41]=0
                    hypothesis=processor.decode(path).split()
                    reference=row['reference'];duration=row['duration']
                    record=dict(utt=row['utt'],wav=row['wav'],duration=duration,reference=reference,
                        text_free=dict(hypothesis=hypothesis,frame_token_ids=path.tolist()),reference_times_used_by_aligner=False)
                    t=totals['text_free'];t['utterances']+=1
                    t['refs'].append(' '.join('d' if p=='dx' else p for p,_,_ in reference if p not in ('sil','q')))
                    t['hyps'].append(' '.join('d' if p=='dx' else p for p in hypothesis if p not in ('sil','q')))
                    changes=np.flatnonzero(path[1:]!=path[:-1])+1
                    score(t,reference,[int(v)*.02 for v in changes],duration)
                    t=totals['forced_k2'];t['utterances']+=1
                    targets=[vocab[p] for p,_,_ in reference]
                    try:
                        if not targets:raise ValueError('No speech-phone reference: FA inapplicable, retained in full-set accounting')
                        segments,k2path=align(logits.log_softmax(-1),targets,duration,backend,blank_policy='previous')
                        spans=[[inverse[s['token_id']],s['start'],s['end']] for s in segments]
                        assert [p for p,_,_ in spans]==[p for p,_,_ in reference]
                        score(t,reference,[v for _,s,e in spans for v in (s,e)],duration)
                        err=t['errors']
                        for (_,rs,re),(_,ps,pe) in zip(reference,spans):
                            err['tokens']+=1;err['start']+=abs(rs-ps);err['end']+=abs(re-pe)
                            err['start_hit20']+=abs(rs-ps)<=.02+1e-9;err['end_hit20']+=abs(re-pe)<=.02+1e-9
                        record['forced_k2']=dict(supplied_target_ids=targets,frame_token_ids=k2path,predicted_spans=spans)
                    except ValueError as exc:
                        t['alignment_failures']+=1;score(t,reference,[],duration)
                        record['forced_k2']=dict(failure=str(exc),supplied_target_ids=targets)
                    f.write(json.dumps(record)+'\n')
                    if (i+1)%200==0 or i+1==len(rows):print('EVALUATE',args.objective,split,i+1,'/',len(rows),flush=True)
            result['splits'][split]={mode:summary(t) for mode,t in totals.items()}
            result.setdefault('data_manifest_sha256',{})[split]=digest(manifest)
    result['label_independence_verified']=independence_checked;result['elapsed_seconds']=time.monotonic()-begin
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    publish(result,output,args.smoke)
    (output/'EVAL_COMPLETE.json').write_text(json.dumps(dict(objective=args.objective,artifact_label=args.label,checkpoint=str(checkpoint),
        smoke=args.smoke,utterances={s:m['forced_k2']['utterances'] for s,m in result['splits'].items()},
        failures={s:m['forced_k2']['alignment_failures'] for s,m in result['splits'].items()}),indent=2)+'\n')
    print('EVAL_COMPLETE',json.dumps(result['splits']),flush=True)


if __name__=='__main__':main()
