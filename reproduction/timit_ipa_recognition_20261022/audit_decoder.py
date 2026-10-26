"""Development-only independent POWSM official greedy-decoder/context checks."""
from pathlib import Path
import importlib.util,json,sys,time,os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'reproduction')]
from reproduction.cif_timit_20261016.data import load_rows,read_audio
from reproduction.timit_external_supplement_20261022.evaluate import sha,write
hp=WORK/'fa-bench/analysis/powsm_ctc/infer.py';sys.path.insert(0,str(hp.parent));spec=importlib.util.spec_from_file_location('powsm_official_decoder_audit_helper',hp);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
@torch.no_grad()
def main():
 assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(0)
 model,cfg=h.load_model(WORK/'fa-bench/data/powsm_ctc_fabench_20261011');model.eval().float();vocab={t:i for i,t in enumerate(model.token_list)}
 from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
 from espnet2.text.token_id_converter import TokenIDConverter
 from espnet2.text.build_tokenizer import build_tokenizer
 official=Speech2TextGreedySearch.__new__(Speech2TextGreedySearch);official.s2t_model=model;official.converter=TokenIDConverter(model.token_list);official.tokenizer=build_tokenizer(token_type=cfg.token_type,bpemodel=cfg.bpemodel)
 rows,_,_=load_rows();byid={r['utt']:r for r in rows['dev']};old={r['utt']:r for r in map(json.loads,(HERE/'powsm/dev.predictions.jsonl').open())}
 selected=[rows['dev'][i] for i in np.linspace(0,399,12,dtype=int)]
 scores=sorted(map(json.loads,(HERE/'powsm/dev.scores.jsonl').open()),key=lambda r:r['phone_errors']/r['reference_phones'],reverse=True)
 for r in scores[:6]:
  if r['utt'] not in {s['utt'] for s in selected}:selected.append(byid[r['utt']])
 prev=torch.tensor([[model.na]],device='cuda');one=torch.tensor([1],device='cuda');started=time.monotonic();records=[]
 for row in selected:
  wave=read_audio(row);n=(len(wave)+639)//640;record=dict(utt=row['utt'],duration=len(wave)/16000,variants={},old_hypothesis=old[row['utt']]['hypothesis'],reference=row['references']['ifmdd_arpabet'])
  for variant,left,lang in [('right20',0,'<unk>'),('left2s',32000,'<unk>'),('left4s',64000,'<unk>'),('eng_prefix',0,'<eng>')]:
   assert left+len(wave)<=320000
   speech=torch.from_numpy(np.pad(wave,(left,320000-left-len(wave)))[None]).cuda();prefix=torch.tensor([[vocab[lang],vocab['<pr>']]],device='cuda')
   enc,lens=model.encode(speech,torch.tensor([320000],device='cuda'),prev.clone(),one,prefix,one*2)
   if isinstance(enc,tuple):enc=enc[0]
   # The official installed decoder reads the entire encoder result and passes IDs through SentencePiece.
   decoded=official._decode_single_sample(enc)[0]
   start=2+left//640;ids=model.ctc.argmax(enc[:,start:start+n])[0].tolist();unique=[];last=None
   for i in ids:
    if i!=last and i!=0:unique.append(i)
    last=i
   tokens=[model.token_list[i] for i in unique if not model.token_list[i].startswith('<')]
   raw=''.join(t.replace('/','').replace('▁','') for t in tokens)
   record['variants'][variant]=dict(hypothesis=raw,acoustic_token_ids=ids,phone_token_count=sum(t.startswith('/') and t.endswith('/') for t in tokens),nonphone_tokens=[t for t in tokens if not(t.startswith('/') and t.endswith('/'))],official_text=decoded[0],official_token_ids=decoded[2])
  record['old_frame_paths_exact_match']=record['variants']['right20']['acoustic_token_ids']==old[row['utt']]['frame_token_ids'];record['old_hypothesis_exact_match']=record['variants']['right20']['hypothesis']==old[row['utt']]['hypothesis'];records.append(record)
  print(row['utt'],record['old_frame_paths_exact_match'],{k:v['hypothesis'] for k,v in record['variants'].items()},flush=True)
 out=HERE/'audit';(out/'decoder_probe.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));write(out/'DECODER_PROBE_COMPLETE.json',dict(status='complete',scope='dev diagnostic cases only; not final benchmark',utterances=len(records),right20_exact_matches=sum(r['old_frame_paths_exact_match'] for r in records),seconds=time.monotonic()-started,pbs_job=os.getenv('PBS_JOBID'),official_decoder_source_sha256=sha(Path(sys.modules['espnet2.bin.s2t_inference_ctc'].__file__))))
 print('DECODER_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
