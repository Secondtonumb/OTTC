"""Local-only pinned CIF epoch24 FA and HuPER TIMIT inference; no training."""
import argparse, ast, hashlib, json, os, platform, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'reproduction')]
from reproduction.cif_timit_20261016.data import load_rows, collate, read_audio
from reproduction.segmentation_metrics import r_value_percent, SOURCE
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def helpers():
 from rapidfuzz.distance import Levenshtein
 ns=dict(Counter=Counter,np=np,Levenshtein=Levenshtein,r_value_percent=r_value_percent)
 for path,names in [(ROOT/'reproduction/libri100_singlehead_20261014/train.py',['phone_per','boundary_matches']),(ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',['edges','new_counts','score','summary'])]:
  nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['cif','huper'],required=True);a=ap.parse_args()
 assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(0)
 out=HERE/a.model;out.mkdir(exist_ok=True);rows,vocab,data=load_rows();h=helpers();begin=time.monotonic()
 if a.model=='cif':
  from ottc.models.sequence.wav2vec2_cif import wvCIFConfig,wvForCIF
  from reproduction.ifmdd_k2_alignment import load_backend,align
  checkpoint=ROOT/'reproduction/cif_fa_probe_20261017/checkpoints/timit_ctc_0p5.pt'
  expected='197da9053fed71a26f466c8d21ac463206034ed9f668fd1dc643fe02d75fc488';assert sha(checkpoint)==expected
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['epoch']==24 and saved['step']==1752
  model=wvForCIF(wvCIFConfig(**saved['model_config']));model.load_state_dict(saved['model'],strict=True);del saved
  backend,k2=load_backend();(out/'ifmdd_k2_source_snapshot.py').write_text(k2.pop('source_snapshot'))
  inv={i:p for p,i in vocab.items()};precision='FP32 standalone';mode='forced_alignment'
 else:
  from transformers import Wav2Vec2Processor,WavLMForCTC
  checkpoint=Path('/work/gm64/m64000/HuPER/artifacts/huper_test_20261021/published/recognizer/model.safetensors')
  expected='6117404b02dcdeeabf3602feae10c2f95f4f5dcf73df8af92a942e301f510113';assert sha(checkpoint)==expected
  processor=Wav2Vec2Processor.from_pretrained(checkpoint.parent,local_files_only=True)
  model,info=WavLMForCTC.from_pretrained(checkpoint.parent,local_files_only=True,output_loading_info=True);assert not any(info.values()),info
  inv={i:processor.tokenizer.convert_ids_to_tokens(i).lower() for i in range(model.config.vocab_size)}
  special=set(processor.tokenizer.all_special_ids);special.add(processor.tokenizer.pad_token_id)
  stride=int(np.prod(model.config.conv_stride));assert stride==320
  mode='text_free';precision='FP32 standalone';k2=None
 model.eval().float().cuda();assert {p.dtype for p in model.parameters()}=={torch.float32}
 source_paths=[Path(__file__),ROOT/'reproduction/cif_timit_20261016/data.py',SOURCE,ROOT/'reproduction/libri100_singlehead_20261014/train.py',ROOT/'reproduction/libri960_continue5_20261017/evaluate.py',ROOT/'reproduction/timit_seed_sweep_20261022/pfer.py']
 prov=dict(model=a.model,checkpoint=str(checkpoint),checkpoint_sha256=expected,data=data,precision=precision,parameter_dtypes=['float32'],selected_epoch=24 if a.model=='cif' else None,selected_step=1752 if a.model=='cif' else None,evaluation_only=True,optimizer_updates=0,evaluation_uploads=False,k2=k2,host=platform.node(),pbs_job=os.getenv('PBS_JOBID'),gpu=torch.cuda.get_device_name(),source_sha256={str(p):sha(p) for p in source_paths})
 write(out/'provenance.json',prov)
 for split in ['dev','test']:
  if (out/f'{split}.COMPLETE.json').exists():continue
  totals=h['new_counts']();f=out/f'{split}.predictions.jsonl';native_counts={str(ms):Counter() for ms in [10,20,30]}
  with f.open('w') as stream:
   for i,row in enumerate(rows[split]):
    ref=[[p,float(s),float(e)] for p,s,e in row['references']['ottc_arpabet']];duration=len(read_audio(row))/16000
    rec=dict(utt=row['utt'],wav=row['wav'],duration=duration,reference=ref,reference_times_used_by_aligner=False)
    totals['utterances']+=1
    if a.model=='cif':
     batch=collate([row],'cuda');hidden,valid=model.encode(batch['input_values'],batch['attention_mask']);frames=int(valid[0].sum())
     logits=model.phone_head.ctc_head(hidden[0,:frames].float()).float();assert torch.isfinite(logits).all()
     targets=[vocab[p] for p,_,_ in ref]
     try:
      segments,path=align(logits.log_softmax(-1),targets,duration,backend,blank_policy='previous');spans=[[inv[s['token_id']],s['start'],s['end']] for s in segments]
      assert [p for p,_,_ in spans]==[p for p,_,_ in ref]
      bounds=[t for _,s,e in spans for t in (s,e)];h['score'](totals,ref,bounds,duration)
      for (_,rs,re),(_,ps,pe) in zip(ref,spans):
       er=totals['errors'];er['tokens']+=1;er['start']+=abs(rs-ps);er['end']+=abs(re-pe);er['start_hit20']+=abs(rs-ps)<=.02+1e-9;er['end_hit20']+=abs(re-pe)<=.02+1e-9
      # Separate CIF-original boundary protocol for within-checkpoint reporting.
      starts=[s for _,s,_ in spans[1:]];refs=[s for _,s,_ in ref[1:]]
      from reproduction.segmentation_metrics import _scorer
      for ms,co in native_counts.items():co.update(_scorer.precision_recall_counts(starts,refs,int(ms)/1000))
      rec['forced_alignment']=dict(predicted_spans=spans,frame_token_ids=path,supplied_target_ids=targets)
     except (ValueError,RuntimeError) as exc:
      totals['alignment_failures']+=1;h['score'](totals,ref,[],duration);rec['forced_alignment']=dict(failure=str(exc))
      from reproduction.segmentation_metrics import _scorer
      for ms,co in native_counts.items():co.update(_scorer.precision_recall_counts([], [s for _,s,_ in ref[1:]],int(ms)/1000))
    else:
     wave=read_audio(row);inp=processor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda')
     logits=model(**inp).logits[0].float();frames=int(model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1))[0]);logits=logits[:frames];assert torch.isfinite(logits).all()
     ids=logits.argmax(-1).tolist();runs=[];start=0
     for t in range(1,len(ids)+1):
      if t==len(ids) or ids[t]!=ids[start]:
       if ids[start] not in special and inv[ids[start]]!='|':runs.append([inv[ids[start]],start*.02,min(t*.02,duration)])
       start=t
     hyp=[p for p,_,_ in runs];bounds=[s for _,s,_ in runs[1:]]
     totals['refs'].append(' '.join('d' if p=='dx' else p for p,_,_ in ref if p not in ['sil','q']))
     totals['hyps'].append(' '.join('d' if p=='dx' else p for p in hyp if p not in ['sil','q']))
     h['score'](totals,ref,bounds,duration);rec['text_free']=dict(hypothesis=hyp,predicted_spans=runs,frame_token_ids=ids)
    rec['frames']=frames;stream.write(json.dumps(rec)+'\n')
    if (i+1)%50==0 or i+1==len(rows[split]):stream.flush();print(a.model,split,i+1,'/',len(rows[split]),'elapsed',round(time.monotonic()-begin,1),flush=True)
  result=dict(utterances=totals['utterances'],model=a.model,split=split,modes={mode:h['summary'](totals)},protocol=dict(boundary_protocol='unique endpoints; exclude utterance edges20ms; chronological one-to-one matching; frozen IF-MDD R-value',inference=precision,clock='duration/frames; blank previous' if a.model=='cif' else 'physical20ms; next nonblank phone onset; internal blank assigned previous',alignment_input='supplied full native phone sequence including sil/q' if a.model=='cif' else 'audio only; no phone count/text/labels input',reference_times_used_by_aligner=False,optimizer_updated=False))
  if a.model=='huper':result['pfer']=score_predictions(f,result);result['modes'][mode]['PFER_percent']=result['pfer']['PFER_percent']
  else:
   from reproduction.segmentation_metrics import _scorer
   result['CIF_original_start_boundary_protocol']={ms:_scorer.finalize_counts(co['precision_matches'],co['recall_matches'],co['predicted_boundaries'],co['reference_boundaries']) for ms,co in native_counts.items()}
  assert len({json.loads(l)['utt'] for l in f.open()})==len(rows[split])
  write(out/f'{split}.json',result);write(out/f'{split}.COMPLETE.json',dict(utterances=len(rows[split]),predictions_sha256=sha(f),metrics_sha256=sha(out/f'{split}.json'),alignment_failures=totals['alignment_failures']))
 assert sha(checkpoint)==expected
 write(out/'COMPLETE.json',dict(status='complete',checkpoint_unchanged=True,dev=400,test=192,seconds=time.monotonic()-begin));print('COMPLETE',a.model,flush=True)
if __name__=='__main__':main()
