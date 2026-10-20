"""Evaluate audio-only decoding, exact IF-MDD K2-FSA and native OT alignment."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT/'reproduction'),str(ROOT)]
import jiwer
import numpy as np
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from multihead import MultiPhoneModel,HEADS
from train_multihead import AudioTargets,file_sha256,greedy,boundary_matches
from alignment_decoders import native_ot_spans
from ifmdd_k2_alignment import load_backend,align as k2_align
from segmentation_metrics import r_value_percent


def accumulator():
    return {'supervised':0,'reference_intervals':0,'alignment_failures':0,'refs':[],'hyps':[],
            'boundaries':{str(ms):Counter() for ms in [10,20,30]},'token_count':0,
            'start_error':0.,'end_error':0.,'start_hit20':0,'end_hit20':0,'iou':0.}


def score_boundaries(counts,reference,predicted,duration):
    ref=sorted({float(t) for _,s,e in reference for t in [s,e] if .02<t<duration-.02})
    pred=sorted(set(float(t) for t in predicted if .02<t<duration-.02))
    counts['reference_intervals']+=1
    for ms,c in counts['boundaries'].items():
        c['matches']+=boundary_matches(ref,pred,int(ms)/1000)
        c['reference']+=len(ref);c['predicted']+=len(pred)


def score_spans(counts,reference,predicted):
    if len(reference)!=len(predicted) or [p for p,_,_ in reference]!=[p for p,_,_ in predicted]:
        raise ValueError('Supplied phone sequence differs from ground truth interval sequence')
    for (_,rs,re),(_,ps,pe) in zip(reference,predicted):
        counts['token_count']+=1
        counts['start_error']+=abs(rs-ps);counts['end_error']+=abs(re-pe)
        counts['start_hit20']+=abs(rs-ps)<=.02+1e-9
        counts['end_hit20']+=abs(re-pe)<=.02+1e-9
        union=max(re,pe)-min(rs,ps)
        counts['iou']+=max(0,min(re,pe)-max(rs,ps))/union if union else 0.


def summarize(counts,mode,head):
    result={'supervised_utterances':counts['supervised'],'utterances_with_reference_intervals':counts['reference_intervals'],
            'alignment_failures':counts['alignment_failures']}
    if mode=='text_free':
        key='token_error_percent' if head=='zipa_ipa' else 'PER_percent'
        result[key]=100*jiwer.process_words(counts['refs'],counts['hyps']).wer if counts['refs'] else None
    if counts['reference_intervals']:
        result['boundaries']={}
        for ms,c in counts['boundaries'].items():
            m,r,p=c['matches'],c['reference'],c['predicted']
            result['boundaries'][ms]={'matches':m,'reference':r,'predicted':p,
                'precision_percent':100*m/p if p else 0.,'recall_percent':100*m/r if r else 0.,
                'f1_percent':200*m/(r+p) if r+p else 0.}
            result['boundaries'][ms]['r_value_percent']=r_value_percent(
                result['boundaries'][ms]['precision_percent'],result['boundaries'][ms]['recall_percent'])
    if counts['token_count']:
        n=counts['token_count']
        result.update(aligned_tokens=n,start_MAE_ms=1000*counts['start_error']/n,
                      end_MAE_ms=1000*counts['end_error']/n,
                      start_hit20_percent=100*counts['start_hit20']/n,end_hit20_percent=100*counts['end_hit20']/n,
                      mean_token_IoU_percent=100*counts['iou']/n)
    return result


def publish(results,model_dir,output):
    from reproduction.wandb_policy import init_evaluation, evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    import wandb
    identity=json.loads((model_dir/'wandb_run.json').read_text())
    identity_path=output/'wandb_run.json'
    previous=json.loads(identity_path.read_text()) if identity_path.exists() else None
    training_config=json.loads((model_dir/'run_config.json').read_text())
    dataset=training_config['dataset']
    # Separate evaluation run preserves completed training histories and step axes.
    label=results.get('comparison_label') or model_dir.name+'-K2FA-vs-text-free'
    options=dict(project=identity['project'],entity=identity['entity'],name=dataset+'-'+label,
                 group=dataset+'_fourphonesets_seed42',job_type='evaluation',tags=['forced-alignment','text-free',model_dir.name],
                 dir=str(ROOT/'reproduction/results/wandb'),mode=os.environ.get('WANDB_MODE','online'),
                 config={'objective':model_dir.name,'training_run':identity['url'],'benchmark':results['protocol'],
                         'checkpoint':results['checkpoint'],'evaluation_only':True,
                         'checkpoint_comparison':results.get('checkpoint_comparison',False)},
                 settings=wandb.Settings(init_timeout=45,disable_code=True))
    if previous:options.update(id=previous['id'],resume='allow')
    try:run=init_evaluation(**options)
    except wandb.errors.CommError:
        options.update(mode='offline');options.pop('resume',None);run=init_evaluation(**options)
    identity_path.write_text(json.dumps({'id':run.id,'url':run.url,'project':identity['project'],
                                         'entity':identity['entity'],'mode':run.settings.mode},indent=2)+'\n')
    values={}
    for split,modes in results['splits'].items():
        for mode,heads in modes.items():
            for head,metrics in heads.items():
                prefix=f'benchmark/{split}/{mode}/{head}'
                for key,value in metrics.items():
                    if key=='boundaries':
                        for ms,c in value.items():
                            for metric,v in c.items():values[f'{prefix}/boundary_{metric}_{ms}ms']=v
                    elif isinstance(value,(int,float)):values[f'{prefix}/{key}']=value
    run.summary.update(values);run.log(values)
    artifact=wandb.Artifact(dataset+'-'+label+'-FA-text-free',type='evaluation')
    for path in output.glob('*.json*'):
        if path.name!='wandb_run.json':artifact.add_file(str(path),name=path.name)
    artifact.add_file(str(output/'ifmdd_k2_source_snapshot.py'),name='ifmdd_k2_source_snapshot.py')
    run.log_artifact(artifact);run.finish()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=['ctc','ce','jsd_upstream'],required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--no-wandb',action='store_true')
    parser.add_argument('--experiment',type=Path,default=ROOT/'reproduction/multihead_timit')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--splits',nargs='+',choices=['dev','test'],default=['dev','test'])
    parser.add_argument('--text-free-only',action='store_true')
    parser.add_argument('--no-native-ot',action='store_true')
    parser.add_argument('--comparison-label')
    args=parser.parse_args();torch.set_num_threads(4)
    assert torch.cuda.is_available(),'Submit evaluation on a GPU compute node'
    directory=args.experiment
    model_dir=directory/'models'/args.objective
    if args.checkpoint is None:
        complete=json.loads((model_dir/'TRAIN_COMPLETE.json').read_text())
        assert complete['epochs']==40 and not complete['smoke']
    elif args.output is None:
        raise ValueError('Custom checkpoint requires a separate output directory')
    checkpoint=args.checkpoint or model_dir/'final'
    model,info=MultiPhoneModel.from_pretrained(checkpoint,output_loading_info=True)
    assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    model=model.eval().cuda()
    output=args.output or directory/'evaluation_modes'/args.objective
    if args.limit:output=output/'validation' if args.output else directory/'validation/evaluation_modes'/args.objective
    output.mkdir(parents=True,exist_ok=True)
    backend,backend_provenance=load_backend()
    source_snapshot=backend_provenance.pop('source_snapshot')
    (output/'ifmdd_k2_source_snapshot.py').write_text(source_snapshot+'\n')
    modes=['text_free'] if args.text_free_only else ['text_free','forced_k2']+(['forced_native_ot'] if args.objective!='ctc' and not args.no_native_ot else [])
    result={'objective':args.objective,'checkpoint':str(checkpoint),'checkpoint_sha256':file_sha256(checkpoint/'pytorch_model.bin'),
            'evaluator_sha256':file_sha256(Path(__file__)),'decoders_sha256':file_sha256(ROOT/'reproduction/alignment_decoders.py'),
            'protocol':{'text_free':'Audio only; greedy frame argmax collapse, including blank transitions for segmentation',
                        'forced_k2':'Exact IF-MDD native-k2 CTC graph/DenseFsaVec/pruned intersection/shortest_path; blank policy previous; no torchaudio fallback',
                        'forced_native_ot':'OTTC heads only: audio frame masses transported to equal target masses; inverse-CDF fractional frame edges',
                        'forced_reference':'Native per-head reference phoneme sequence; not word-text G2P or an acoustic MFA alignment',
                        'frame_stride_seconds':.02,'time_shift_seconds':0,'edge_exclusion_seconds':.02,
                        'IPA_segmentation':'No real MFA/ZIPA reference intervals; export alignments without boundary-accuracy claims',
                        'recognition':'Only text_free reports PER/token error; forced-alignment labels supplied by reference',
                        'k2_forced_spans':'Exact IF-MDD timestamp conversion: blank gaps assigned to previous token; duration / frames',
                        'k2_backend':backend_provenance},
            'splits':{},'replay':{},'label_independence_verified':False,
            'checkpoint_comparison':args.checkpoint is not None,'comparison_label':args.comparison_label,
            'data_manifest_sha256':{s:file_sha256(directory/'data'/('train-dev.jsonl' if s=='dev' else 'test.jsonl')) for s in args.splits}}
    begin=time.monotonic()
    inverse={h:{v:k for k,v in model.config.vocabs[h].items()} for h in HEADS}
    with torch.inference_mode():
        for split,file in [('dev','train-dev.jsonl'),('test','test.jsonl')]:
            if split not in args.splits:continue
            data=AudioTargets(directory/'data'/file,args.limit)
            prior=None if args.checkpoint else {r['utt']:r for r in (json.loads(line) for line in (model_dir/(split+'_predictions.jsonl')).open())}
            totals={mode:{h:accumulator() for h in HEADS} for mode in modes}
            replay=Counter()
            with (output/(split+'_predictions.jsonl')).open('w') as f:
                for i in range(len(data)):
                    sample=data[i];record=sample['record']
                    values=torch.from_numpy(sample['input_values']).unsqueeze(0).cuda()
                    mask=torch.ones_like(values,dtype=torch.long)
                    # Inference is explicitly audio-only for every head.
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred=model(input_values=values,attention_mask=mask)
                    length=int(pred['lengths'][0]);duration=record['duration']
                    if not result['label_independence_verified']:
                        labels={h:torch.tensor([record['targets'][h]],device='cuda',dtype=torch.long) for h in HEADS}
                        with torch.autocast('cuda',dtype=torch.bfloat16):check=model(values,mask,labels)
                        for h in HEADS:torch.testing.assert_close(pred['logits'][h],check['logits'][h],rtol=0,atol=0)
                        result['label_independence_verified']=True
                    masses=None
                    if 'forced_native_ot' in modes:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            mass_hidden=model.alignment_encoder(values,attention_mask=mask).last_hidden_state
                            masses={h:model.mass_heads[h](mass_hidden)[0,:length,0].float().softmax(-1).cpu().numpy() for h in HEADS}
                    example={'utt':record['utt'],'wav':record['wav'],'duration':duration,'modes':{mode:{} for mode in modes}}
                    for h in HEADS:
                        logits=pred['logits'][h][0,:length].float()
                        path=logits.argmax(-1).cpu().tolist();tokens=greedy(path)
                        if prior is not None:
                            replay['heads_checked']+=1
                            replay['identical_frame_paths']+=path==prior[record['utt']]['heads'][h]['frame_token_ids']
                        t=totals['text_free'][h]
                        targets=record['targets'][h]
                        reference=record['references'].get(h,[])
                        hyp=[inverse[h][v] for v in tokens]
                        example['modes']['text_free'][h]={'predicted':hyp,'frame_token_ids':path,'audio_only':True}
                        if targets:
                            t['supervised']+=1;t['refs'].append(' '.join(str(v) for v in targets));t['hyps'].append(' '.join(map(str,tokens)))
                        if reference:
                            boundaries=[i*.02 for i in range(1,len(path)) if path[i]!=path[i-1]]
                            score_boundaries(t,reference,boundaries,duration)
                        if not targets:
                            for mode in modes[1:]:example['modes'][mode][h]={'skipped':'No complete reference phone sequence'}
                            continue
                        for mode in modes[1:]:
                            t=totals[mode][h];t['supervised']+=1
                            try:
                                if mode=='forced_k2':
                                    aligned,k2_path=k2_align(logits.log_softmax(-1),targets,duration,backend)
                                    spans=[(inverse[h][s['token_id']],s['start'],s['end']) for s in aligned]
                                else:
                                    spans=native_ot_spans(masses[h],targets)
                                    spans=[(inverse[h][v],s*.02,min(e*.02,duration)) for v,s,e in spans]
                                example['modes'][mode][h]={'supplied_phones':[inverse[h][v] for v in targets],
                                                          'spans_seconds':spans,'reference_times_used':False}
                                if mode=='forced_k2':example['modes'][mode][h]['frame_token_ids']=k2_path
                                if reference:
                                    score_boundaries(t,reference,[v for _,s,e in spans for v in [s,e]],duration)
                                    score_spans(t,reference,spans)
                            except ValueError as exc:
                                t['alignment_failures']+=1
                                example['modes'][mode][h]={'failure':str(exc)}
                    f.write(json.dumps(example,ensure_ascii=False)+'\n')
                    if (i+1)%100==0:print(args.objective,split,i+1,'/',len(data),flush=True)
            result['splits'][split]={mode:{h:summarize(t,mode,h) for h,t in heads.items()} for mode,heads in totals.items()}
            result['replay'][split]=dict(replay)
            # Exact equality of frame predictions proves that separating modes
            # preserved the existing text-free benchmark.
            assert replay['identical_frame_paths']==replay['heads_checked'],dict(replay)
            for mode in modes[1:]:
                assert not any(t['alignment_failures'] for t in totals[mode].values()),mode
    result['elapsed_seconds']=time.monotonic()-begin
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not args.no_wandb:publish(result,model_dir,output)
    (output/'EVAL_COMPLETE.json').write_text(json.dumps({'objective':args.objective,'limit':args.limit,
                                                       'modes':modes,'checkpoint':str(checkpoint)},indent=2)+'\n')
    if not args.limit and not args.checkpoint:
        from english_suite.summarize_results import main as summarize_results
        summarize_results()
    print(json.dumps(result['splits'].get('test',result['splits'].get('dev')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
