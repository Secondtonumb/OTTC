"""Pinned ZIPA/POWSM audio-only TIMIT greedy CTC predictions, local-only."""
import argparse,json,sys,time,os,platform,importlib.util,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'reproduction')]
from reproduction.cif_timit_20261016.data import load_rows,read_audio
from reproduction.timit_external_supplement_20261022.evaluate import sha,write
VHELP=WORK/'dataset/voxangeles/evaluation_20261021/infer.py'
def imp(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
@torch.no_grad()
def main():
 a=argparse.ArgumentParser();a.add_argument('model',choices=['zipa','powsm']);args=a.parse_args();out=HERE/(args.model+'_eng');out.mkdir(exist_ok=True)
 assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(0)
 rows,_,data=load_rows();vh=imp('vox_infer_primitives',VHELP);started=time.monotonic();sources=[Path(__file__),VHELP]
 if args.model=='zipa':
  hp=WORK/'IF-MDD/zipa_try/eval_zipa_cmu_monophone.py';h=imp('zipa_timit_recognition_helper',hp);checkpoint=WORK/'IF-MDD/zipa_try/zipa-large-crctc-ns-800k/zipa_large_crctc_0.5_scale_800000_avg10.pth';assert sha(checkpoint)=='dcf8d4b507d0eacbac40735dd742a4c85071c52aab35255cc27da644f68b8584'
  bpe=WORK/'zipa/ipa_simplified/unigram_127.model';tokens=checkpoint.parent/'tokens.txt';runner=h.ZipATorchRunner(checkpoint,bpe,'cuda','large').load();labels=h.load_tokens(tokens);model=runner.model;sources+=[hp,checkpoint,bpe,tokens];frame=.02
  def forward(wave):
   p,n=runner.encode_batch([torch.from_numpy(wave)]);return p[0,:n[0]],h.SPECIAL_LABELS
 else:
  hp=WORK/'fa-bench/analysis/powsm_ctc/infer.py';sys.path.insert(0,str(hp.parent));h=imp('powsm_timit_recognition_helper',hp);run=WORK/'fa-bench/data/powsm_ctc_fabench_20261011';checkpoint=run/'model/model.pth';assert sha(checkpoint)=='d9d56a166e5dfe3088ca2080c8c3e2eb8124827166a68cd41b673963f99c9ef8'
  model,cfg=h.load_model(run);assert model.blank_id==0;labels={i:s.replace('/','').replace('▁','') for i,s in enumerate(model.token_list)};vocab={s:i for i,s in enumerate(model.token_list)};prefix=torch.tensor([[vocab['<eng>'],vocab['<pr>']]],device='cuda');prev=torch.tensor([[model.na]],device='cuda');one=torch.tensor([1],device='cuda');special={labels[i] for i,s in enumerate(model.token_list) if s.startswith('<') or not labels[i]};sources+=[hp,checkpoint,run/'model/config.yaml',hp.parent/'vendor/subsampling.py'];frame=.04
  def forward(wave):
   assert len(wave)<=320000,'No truncation allowed'
   speech=torch.from_numpy(np.pad(wave,(0,320000-len(wave)))[None]).cuda();enc,lens=model.encode(speech,torch.tensor([320000],device='cuda'),prev,one,prefix,one*2)
   if isinstance(enc,tuple):enc=enc[0]
   frames=(len(wave)+639)//640;assert enc.shape[1]>=frames+2
   return model.ctc.log_softmax(enc[:,2:2+frames])[0],special
 model.eval().float();assert {p.dtype for p in model.parameters()}=={torch.float32}
 prov=dict(model=args.model,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),data=data,parameters='FP32',inference='FP32 batch1 greedy CTC; collapse repeats then remove blank/specials',audio_only=True,language_symbol='<eng>',language_input='known English corpus metadata; no reference transcript',reference_text_used=False,reference_phones_used=False,frame_seconds=frame,powsm_prefix_removed=2 if args.model=='powsm' else 0,host=platform.node(),gpu=torch.cuda.get_device_name(),pbs_job=os.getenv('PBS_JOBID'),evaluation_uploads=False,source_sha256={str(p):sha(p) for p in sources})
 write(out/'provenance.json',prov)
 for split,expected in [('dev',400),('test',192)]:
  if (out/f'{split}.INFERENCE_COMPLETE.json').exists():continue
  f=out/f'{split}.predictions.jsonl'
  with f.open('w') as stream:
   for i,row in enumerate(rows[split]):
    wave=read_audio(row);duration=len(wave)/16000;probs,special=forward(wave);assert torch.isfinite(probs).all();ids=probs.argmax(-1).cpu().tolist();spans=vh.ctc_segments(ids,labels,0,frame,duration,special)
    rec=dict(utt=row['utt'],model=args.model,duration=duration,wav=row['wav'],hypothesis=''.join(s['label'] for s in spans),raw_segments=spans,frame_token_ids=ids,reference=row['references']['ifmdd_arpabet'],reference_text_used=False)
    stream.write(json.dumps(rec,ensure_ascii=False)+'\n')
    if (i+1)%50==0 or i+1==expected:stream.flush();print(args.model,split,i+1,'/',expected,'seconds',round(time.monotonic()-started,1),flush=True)
  assert len({json.loads(l)['utt'] for l in f.open()})==expected
  write(out/f'{split}.INFERENCE_COMPLETE.json',dict(utterances=expected,predictions_sha256=sha(f)))
 assert sha(checkpoint)==prov['checkpoint_sha256'];write(out/'INFERENCE_COMPLETE.json',dict(status='complete',dev=400,test=192,checkpoint_unchanged=True,seconds=time.monotonic()-started));print('COMPLETE',args.model,flush=True)
if __name__=='__main__':main()
