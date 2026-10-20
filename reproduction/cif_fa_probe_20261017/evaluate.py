"""Bounded CIF length-conditioned and auxiliary-head K2 alignment evaluation."""
from collections import Counter
from datetime import datetime
import ast,gc,hashlib,importlib.util,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from ottc.models.sequence.wav2vec2_cif import wvCIFConfig,wvForCIF
from reproduction.ifmdd_k2_alignment import align,load_backend
from reproduction.segmentation_metrics import _scorer,SOURCE

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def write(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def scored(row,bounds):
 bounds=sorted(set(float(t) for t in bounds if .02<t<row['duration']-.02))
 reference=row['reference']['boundaries'];metrics={}
 for ms in [10,20,25]:
  hit=boundary_matches(reference,bounds,ms/1000)
  metrics[str(ms)]=_scorer.finalize_counts(hit,hit,len(bounds),len(reference))
 return bounds,metrics


def span_record(row,job,method,spans,bounds,**extra):
 bounds,metrics=scored(row,bounds)
 assert all(0<=a<b<=row['duration']+1e-6 for _,a,b in spans)
 assert all(a[2]<=b[1]+1e-7 for a,b in zip(spans,spans[1:]))
 return dict(recording_id=row['recording_id'],utt=row['utt'],dataset=row['dataset'],split=row['split'],
  model_label=job['label'],job_key=job['key'],method=method,scope='four preselected visualization examples only',
  duration=row['duration'],phones=[p for p,_,_ in spans],pred_starts=[a for _,a,_ in spans],pred_ends=[b for _,_,b in spans],
  pred_boundaries=bounds,metrics=metrics,text_free=method=='text_free',reference_times_used_by_aligner=False,
  supplied_phones=row['supplied_phones'] if method!='text_free' else None,
  supplied_token_ids=row['supplied_token_ids'] if method=='auxiliary_k2' else None,
  native_training_target_count=row['native_training_target_count'],removed_edge_sil=row['removed_edge_sil'],
  phone_mapping_note='native dx retained; common-reference consistency maps dx→d. Libri edge sil targets removed to match existing plotted targets.',**extra)


@torch.no_grad()
def main():
 assert torch.cuda.is_available(),'Use a PBS GPU allocation for the real models'
 torch.set_num_threads(4)
 global boundary_matches
 matcher=ROOT/'reproduction/buckeye_singlehead_20261014/train.py'
 tree=ast.parse(matcher.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='boundary_matches')
 ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(matcher),'exec'),ns)
 boundary_matches=ns['boundary_matches']
 inputs=json.loads((HERE/'INPUTS.json').read_text())
 backend,provenance=load_backend()
 (HERE/'ifmdd_k2_source_snapshot.py').write_text(provenance.pop('source_snapshot')+'\n')
 sources={str(ROOT/'ottc/models/sequence/cif.py'):digest(ROOT/'ottc/models/sequence/cif.py'),
          str(ROOT/'ottc/models/sequence/wav2vec2_cif.py'):digest(ROOT/'ottc/models/sequence/wav2vec2_cif.py'),
          str(SOURCE):digest(SOURCE),str(Path(__file__)):digest(__file__),str(HERE/'INPUTS.json'):digest(HERE/'INPUTS.json')}
 scorer_dir=ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source'
 sys.path.insert(0,str(scorer_dir))
 scorer_file=scorer_dir/'recompute_timit_raw_phn_tse_rvalue.py'
 spec=importlib.util.spec_from_file_location('frozen_raw_phn',scorer_file)
 raw_scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw_scorer)
 sources[str(scorer_file)]=digest(scorer_file)
 phn=Path('/work/gm64/m64000/dataset/TIMIT/timit/test/dr7/fdhc0/si1559.phn')
 sources[str(phn)]=digest(phn)
 all_results=[];job_results=[];started=time.monotonic()
 for job in inputs['jobs']:
  output=HERE/'runs'/job['key'];output.mkdir(parents=True,exist_ok=True)
  complete=output/'EVAL_COMPLETE.json'
  if complete.exists():
   all_results.extend(json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines())
   job_results.append(json.loads((output/'metrics.json').read_text()));continue
  before=Path(job['checkpoint']).stat();checkpoint_hash=digest(job['checkpoint'])
  checkpoint=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)
  assert checkpoint['step']==job['expected_step'],(job['key'],checkpoint['step'],job['expected_step'])
  for f,h in job['original_training_contract']['sources_sha256'].items():
   if f in ['ottc/models/sequence/cif.py','ottc/models/sequence/wav2vec2_cif.py']:
    assert digest(ROOT/f)==h,('trained source changed',f)
  config=checkpoint['model_config'];assert config['quantity_weight']==1
  model=wvForCIF(wvCIFConfig(**config));model.load_state_dict(checkpoint['model'],strict=True)
  model.eval().cuda();selected_epoch,selected_step=checkpoint['epoch'],checkpoint['step'];del checkpoint
  results=[];timing=[];counts={};torch.cuda.reset_peak_memory_stats()
  for row in [r for r in inputs['inputs'] if r['dataset']==job['dataset']]:
   wave,sr=sf.read(row['wav'],dtype='float32');assert sr==16000 and wave.ndim==1
   assert len(wave)/sr==row['duration']
   wave=(wave-wave.mean())/np.sqrt(wave.var()+1e-7)
   values=torch.from_numpy(wave)[None].cuda();mask=torch.ones_like(values,dtype=torch.long)
   torch.cuda.synchronize();t0=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    hidden,valid=model.encode(values,mask)
    free=model.phone_head.cif(hidden,valid)
    supplied_length=torch.tensor([len(row['supplied_phones'])],device='cuda',dtype=torch.long)
    conditional=model.phone_head.cif(hidden,valid,supplied_length)
   torch.cuda.synchronize();t1=time.monotonic()
   frames=int(valid[0].sum());duration=row['duration'];n=int(conditional.lengths[0]);assert n==len(row['supplied_phones'])
   assert torch.equal(free.quantity,conditional.quantity),'Quantity must remain unscaled'
   assert abs(float(conditional.weights[0,:frames].sum())-n)<1e-3
   positions=conditional.fire_positions[0,:n-1].float().cpu().tolist()
   bounds=[p*duration/frames for p in positions]
   edges=[0.]+bounds+[duration]
   native=span_record(row,job,'cif_length_conditioned',list(zip(row['supplied_phones'],edges,edges[1:])),bounds,
    raw_quantity=float(free.quantity[0]),free_fired_tokens=int(free.lengths[0]),supplied_length=n,frame_count=frames,
    uses_phone_identities_for_boundary_search=False,clock='CIF fractional fire positions × duration/valid_frames',
    alignment_input='reference_phone_count_only',description='CIF reference-length scaling; no K2 and no phone-identity boundary search')
   # Re-score the saved audio-only path on the same grid, without claiming a
   # new recognition decoding or a reference-conditioned PER.
   free_n=int(free.lengths[0]);free_bounds=(free.fire_positions[0,:max(free_n-1,0)].float()*duration/frames).cpu().tolist()
   native['text_free_boundary_baseline']=dict(fired_tokens=free_n,metrics=scored(row,free_bounds)[1],pred_boundaries=scored(row,free_bounds)[0])
   results.append(native)
   head=model.phone_head.ctc_head if model.phone_head.ctc_head is not None else model.phone_head.ottc_head
   if head is not None:
    with torch.autocast('cuda',enabled=False):log_probs=head(hidden[0,:frames].float()).log_softmax(-1)
    assert log_probs.shape==(frames,config['vocab_size']) and torch.isfinite(log_probs).all()
    try:
     segments,path=align(log_probs,row['supplied_token_ids'],duration,backend,blank_policy='previous')
     inverse={int(i):p for p,i in inputs['vocab'].items()}
     spans=[[inverse[s['token_id']],s['start'],s['end']] for s in segments]
     assert [p for p,_,_ in spans]==row['supplied_phones']
     k2=span_record(row,job,'auxiliary_k2',spans,[t for _,a,b in spans for t in(a,b)],
      frame_count=frames,frame_token_ids=path,auxiliary_head='ctc' if model.phone_head.ctc_head is not None else 'ottc',
      uses_phone_identities_for_boundary_search=True,clock='saved IF-MDD K2 duration/frames; blank previous',
      alignment_input='supplied_phone_sequence',description='K2 CTC graph over auxiliary encoder-frame posterior; not CIF firing boundaries')
     results.append(k2)
    except (ValueError,RuntimeError) as exc:
     results.append(dict(recording_id=row['recording_id'],job_key=job['key'],model_label=job['label'],
        method='auxiliary_k2',alignment_failed=True,error=str(exc),scope='selected example only'))
   torch.cuda.synchronize();timing.append(dict(recording_id=row['recording_id'],encode_and_cif_seconds=t1-t0,
      total_seconds=time.monotonic()-t0,frame_count=frames,raw_quantity=float(free.quantity[0])))
   print('EVALUATED',job['key'],row['recording_id'],'length',n,'free',int(free.lengths[0]),flush=True)
  for result in results:
   if result.get('alignment_failed'):continue
   result.update(checkpoint=job['checkpoint'],checkpoint_sha256=checkpoint_hash,selected_epoch=selected_epoch,selected_step=selected_step)
   if result['dataset']=='timit':
    prediction=dict(utt=result['utt'],phones=result['phones'],pred_phones=result['phones'],pred_starts=result['pred_starts'],pred_ends=result['pred_ends'])
    ref=raw_scorer.load_raw_phn_ref({('FDHC0','SI1559'):phn},result['utt'],sample_rate=16000.,dx_map='d',closure_mode='silence')
    pred=raw_scorer.prediction_rows(prediction,dx_map='d',hyp_round_step_ms=0.)
    raw,_=raw_scorer.evaluate_pair(pred,ref,exclude_length_mismatch=True,filter_silence_for_metrics=True,ignore_edge_boundaries_for_tse_acc=False)
    result['timit_raw_PHN_separate']=dict(scope='single selected TIMIT example',metrics=raw,common_TSE_comparison_performed=False)
   for ms,m in result['metrics'].items():
    c=counts.setdefault(result['method'],{}).setdefault(ms,Counter())
    c.update({k:m[k] for k in ['matches','predicted_boundaries','reference_boundaries']})
  summary=dict(job_key=job['key'],dataset=job['dataset'],variant=job['variant'],model_label=job['label'],
   scope='bounded four-example visualization probe; not a full benchmark',evaluation_only=True,optimizer_updates=0,
   selected_epoch=selected_epoch,selected_step=selected_step,checkpoint_sha256=checkpoint_hash,checkpoint=job['checkpoint'],
   utterances=len(timing),alignment_failures=sum(bool(r.get('alignment_failed')) for r in results),
   methods={method:{ms:_scorer.finalize_counts(c['matches'],c['matches'],c['predicted_boundaries'],c['reference_boundaries']) for ms,c in by_ms.items()} for method,by_ms in counts.items()},
   timing=timing,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,source_sha256=sources,k2=provenance,
   input_protocol='native phone symbols equivalent to common plotted reference after dx→d; remove only Libri edge sil targets; no reference times input')
  (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
  write(output/'metrics.json',summary)
  after=Path(job['checkpoint']).stat()
  assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert digest(job['checkpoint'])==checkpoint_hash
  write(complete,dict(status='complete',job_key=job['key'],utterances=len(timing),records=len(results),alignment_failures=summary['alignment_failures'],checkpoint_unchanged=True))
  all_results.extend(results);job_results.append(summary)
  del model,hidden,free,conditional;gc.collect();torch.cuda.empty_cache()
 (HERE/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_results))
 write(HERE/'RESULTS.json',dict(scope='four preselected visualization utterances; nine matching trained checkpoints',jobs=job_results,records=len(all_results)))
 write(HERE/'EVAL_COMPLETE.json',dict(status='complete',scope='bounded four-example probe only',jobs=len(job_results),
  utterances=4,model_utterance_pairs=12,records=len(all_results),expected_records=20,
  alignment_failures=sum(j['alignment_failures'] for j in job_results),wall_seconds=time.monotonic()-started,
  host=platform.node(),pbs_job=os.environ.get('PBS_JOBID'),finished_at=datetime.now().astimezone().isoformat(),new_training=False,k2=provenance,source_sha256=sources))
 assert len(all_results)==20
 print('COMPLETE',HERE/'EVAL_COMPLETE.json',flush=True)

if __name__=='__main__':main()
