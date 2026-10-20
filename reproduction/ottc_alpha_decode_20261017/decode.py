"""Audio-only alpha-mass decoding. Does not read scoring reference labels/times."""
import gc,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from transformers import Wav2Vec2Processor
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from reproduction.alignment_decoders import native_ot_spans

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


@torch.no_grad()
def main():
 assert torch.cuda.is_available(),'Use allocated GH200'
 torch.set_num_threads(4)
 if (HERE/'DECODE_COMPLETE.json').exists():print('Reuse completed decoding');return
 inputs=json.loads((HERE/'INPUTS.json').read_text())
 results=[];models=[];started=time.monotonic()
 source_paths=[Path(__file__),ROOT/'reproduction/alignment_decoders.py',ROOT/'ottc/models/sequence/wav2vec2_ottc.py',HERE/'INPUTS.json']
 sources={str(p):digest(p) for p in source_paths}
 for job in inputs['jobs']:
  checkpoint=Path(job['checkpoint']);weights=checkpoint/'model.safetensors'
  before=weights.stat();assert digest(weights)==job['checkpoint_sha256']
  model,info=wvForOTTC.from_pretrained(checkpoint,output_loading_info=True)
  assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
  assert model.config.bins_metric=='jsd','Must use original upstream JSD'
  model.step=int(checkpoint.name.split('-')[-1]);model.eval().cuda()
  directory=checkpoint.parent
  processor=Wav2Vec2Processor.from_pretrained(directory/'processor')
  vocab=json.loads((directory/'vocab_used.json').read_text());inverse={int(v):p for p,v in vocab.items()}
  state=json.loads((checkpoint/'trainer_state.json').read_text())
  torch.cuda.reset_peak_memory_stats();t0=time.monotonic()
  for case in job['cases']:
   assert digest(case['wav'])==case['wav_sha256']
   wave,sr=sf.read(case['wav'],dtype='float32');assert sr==16000 and wave.ndim==1
   batch=processor(wave,sampling_rate=sr,return_tensors='pt',return_attention_mask=True).to('cuda')
   # Same FP32 audio-only forward as the previous single-head evaluation.
   out=model(**batch)
   frames=int(out.hidden_states[1][0].sum());logits=out.logits[0,:frames].float().cpu().numpy()
   alpha=out.hidden_states[0][0,:frames].float().cpu().numpy().astype(np.float64)
   assert np.isfinite(alpha).all() and np.all(alpha>=0) and alpha.sum()>0
   ids=logits.argmax(-1);ids[alpha==0]=0;ids[ids==41]=0
   hyp=processor.decode(ids).split();target_ids=[vocab[p] for p in hyp]
   assert target_ids and all(v not in (0,41) for v in target_ids)
   prior=case['prior_audio_only_prediction']
   assert hyp==prior['hypothesis'],('Recognition replay mismatch',case['recording_id'])
   if 'frame_token_ids' in prior:assert ids.tolist()==prior['frame_token_ids']
   changes=np.flatnonzero(ids[1:]!=ids[:-1])+1
   starts,ends=np.r_[0,changes],np.r_[changes,len(ids)]
   baseline=[[inverse[int(ids[s])],float(s)*.02,min(float(e)*.02,case['duration'])] for s,e in zip(starts,ends)]
   if 'spans' in prior:
    assert [x for x in baseline if x[0]!='<pad>']==prior['spans']
    assert [int(i)*.02 for i in changes if .02<int(i)*.02<case['duration']-.02]==prior['boundaries']
   spans=native_ot_spans(alpha,target_ids)
   mass=alpha/alpha.sum();A=np.r_[0.,mass.cumsum()];A[-1]=1.
   U=len(target_ids);B=np.linspace(0.,1.,U+1)
   gamma=np.maximum(0.,np.minimum(A[1:,None],B[None,1:])-np.maximum(A[:-1,None],B[None,:-1]))
   np.testing.assert_allclose(gamma.sum(1),mass,rtol=0,atol=1e-12)
   np.testing.assert_allclose(gamma.sum(0),np.ones(U)/U,rtol=0,atol=1e-12)
   frame_bounds=[s[2] for s in spans[:-1]]
   for j,position in enumerate(frame_bounds,1):
    i=min(int(np.floor(position)),frames-1);fraction=position-i
    np.testing.assert_allclose(A[i]+fraction*mass[i],j/U,atol=1e-12,rtol=0)
   native=[[inverse[token],start*.02,end*.02] for token,start,end in spans]
   grid_frames=np.floor(np.asarray(frame_bounds)+.5)
   shifts=(np.asarray(frame_bounds)-grid_frames)*20
   result=dict(recording_id=case['recording_id'],utt=case['utt'],split=case['split'],dataset=job['dataset'],duration=case['duration'],
    checkpoint=job['checkpoint'],checkpoint_sha256=job['checkpoint_sha256'],original_epoch=state['epoch'],original_step=state['global_step'],
    hypothesis=hyp,hypothesis_token_ids=target_ids,predicted_phone_count=U,frame_count=frames,alpha=mass.tolist(),
    frame_log_probs=(logits-np.logaddexp.reduce(logits,axis=-1)[:,None]).tolist(),frame_token_ids=ids.tolist(),
    transport_gamma=gamma.tolist(),argmax_spans=baseline,argmax_boundaries=(changes*.02).tolist(),
    native_alpha_spans=native,native_alpha_boundaries=(np.asarray(frame_bounds)*.02).tolist(),
    native_alpha_grid_boundaries=(grid_frames*.02).tolist(),fractional_boundary_frame_positions=frame_bounds,
    subframe_shift_from_nearest_grid_ms=shifts.tolist(),mean_abs_subframe_shift_ms=float(np.abs(shifts).mean()),
    max_abs_subframe_shift_ms=float(np.abs(shifts).max()),
    text_free=True,reference_phone_count_used=False,reference_phone_labels_used=False,reference_times_used=False,
    recognition_replay_identical=True,transport_marginals_verified=True,
    beta_rule='uniform 1/predicted_phone_count from audio-only greedy hypothesis; no reference count or labels',
    clock='fixed20ms frame cells for argmax, fractional alpha and rounded alpha; no fitted shift; final display tail unscored',
    scope='one of four preselected examples, not full benchmark')
   results.append(result)
   print('DECODED',case['recording_id'],'predicted phones',U,'mean subframe shift ms',result['mean_abs_subframe_shift_ms'],flush=True)
  torch.cuda.synchronize()
  after=weights.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert digest(weights)==job['checkpoint_sha256']
  models.append(dict(dataset=job['dataset'],checkpoint=job['checkpoint'],checkpoint_sha256=job['checkpoint_sha256'],
   original_epoch=state['epoch'],original_step=state['global_step'],utterances=len(job['cases']),loading_info=info,
   checkpoint_unchanged=True,seconds=time.monotonic()-t0,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
  del model,out,batch;gc.collect();torch.cuda.empty_cache()
 assert len(results)==4
 (HERE/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
 write(HERE/'DECODE_COMPLETE.json',dict(status='complete',scope='four preselected examples only',utterances=4,models=models,
  source_sha256=sources,reference_inputs_used=False,new_training=False,wall_seconds=time.monotonic()-started,
  host=platform.node(),pbs_job=os.environ.get('PBS_JOBID'),recognition_replay_identical=True,transport_marginals_verified=True))
 print('DECODE COMPLETE',flush=True)

if __name__=='__main__':main()
