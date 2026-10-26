"""Re-run pinned ZIPA K2 FA; audit peak boundaries, blank policy and time clock."""
import sys,json,hashlib,importlib.util,time,os,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'reproduction')]
from reproduction.timit_external_supplement_20261022.evaluate import helpers,sha,write
from reproduction.cif_timit_20261016.data import load_rows,read_audio
HELPER=WORK/'IF-MDD/zipa_try/eval_zipa_cmu_monophone.py'
CKPT=WORK/'IF-MDD/zipa_try/zipa-large-crctc-ns-800k/zipa_large_crctc_0.5_scale_800000_avg10.pth'
TOKENS=CKPT.parent/'tokens.txt';BPE=WORK/'zipa/ipa_simplified/unigram_127.model'
OLD=WORK/'IF-MDD/fa_research/results/zipa_large_crctc_ns_800k_cmu_monophone_20260624/zipa_large_crctc_ns_800k'
@torch.no_grad()
def main():
 assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(0)
 assert sha(CKPT)=='dcf8d4b507d0eacbac40735dd742a4c85071c52aab35255cc27da644f68b8584'
 spec=importlib.util.spec_from_file_location('zipa_k2_audit_helper',HELPER);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
 runner=h.ZipATorchRunner(CKPT,BPE,'cuda','large').load();assert {p.dtype for p in runner.model.parameters()}=={torch.float32}
 vocab=h.load_tokens(TOKENS);lookup=h.build_token_lookup(vocab);beam=h.parse_native_k2_beam_configs('80:40:200:100000,160:80:500:300000,320:160:1000:1000000,640:320:2000:2000000')
 funcs=helpers();rows,_,data=load_rows();started=time.monotonic()
 (HERE/'zipa_helper_snapshot.py').write_bytes(HELPER.read_bytes())
 source={str(p):sha(p) for p in [Path(__file__),HELPER,TOKENS,BPE,CKPT,ROOT/'reproduction/ifmdd_k2_alignment.py']}
 provenance=dict(checkpoint=str(CKPT),checkpoint_sha256=sha(CKPT),model='ZIPA Large CR-CTC-NS 800k Avg10',parameter_dtypes=['float32'],compute='FP32',data=data,source_sha256=source,alignment='native k2.ctc_graph modified=False; intersect_dense_pruned; shortest_path',graph_input='reference phones mapped to IPA; no reference times',mapping='existing helper AH→ə; diphthongs expanded then regrouped; common reference removes sil/q because no silence/glottal target token',primary='previous blank policy; physical20ms frame clock; speech-only MAE',host=platform.node(),pbs_job=os.getenv('PBS_JOBID'),gpu=torch.cuda.get_device_name(),evaluation_uploads=False)
 write(HERE/'provenance.json',provenance)
 all_metrics={}
 for split,expected in [('dev',400),('test',192)]:
  legacy_path=OLD/('timit_'+split)/'forced_alignment/alignments.jsonl';legacy={r['utt']:r for r in map(json.loads,legacy_path.open())};assert len(legacy)==expected
  counters={(ref,policy,clock):funcs['new_counts']() for ref in ['legacy','common'] for policy in ['previous','drop'] for clock in ['physical20ms','duration_per_frame']}
  metrics_file=HERE/f'{split}.json';pred_file=HERE/f'{split}.predictions.jsonl';agreements=[];peak_count=peak_frames=total_frames=blank_frames=0
  with pred_file.open('w') as stream:
   for idx,row in enumerate(rows[split]):
    audio=read_audio(row);duration=len(audio)/16000;matrix,lens=runner.encode_batch([torch.from_numpy(audio)]);matrix=matrix[0,:lens[0]].clone().float();n=int(lens[0]);assert torch.isfinite(matrix).all();total_frames+=n;blank_frames+=int((matrix.argmax(-1)==0).sum())
    record=dict(utt=row['utt'],wav=row['wav'],duration=duration,frames=n,effective_historical_frame_ms=1000*duration/n,reference_times_used_by_aligner=False,alignments={})
    for ref_type in ['legacy','common']:
     if ref_type=='legacy':
      old=legacy[row['utt']];ref=list(zip(old['gt_phones'],old['gt_starts'],old['gt_ends']))
     else:ref=[(p,float(s),float(e)) for p,s,e in row['references']['ifmdd_arpabet'] if p not in ['sil','q']]
     phones=[p for p,_,_ in ref];targets,sizes,ipa=h.reference_to_ipa_targets(phones,lookup,'ə')
     assert n>=h.ctc_min_frames(targets)
     with torch.inference_mode(False):
      labels,scores=h.native_k2_forced_align_ids(matrix[None].clone(),targets,0,beam)
     collapsed=[];prev=None
     for token in labels.tolist():
      if token and token!=prev:collapsed.append(token)
      prev=token
     assert collapsed==targets
     tokens=h.dense_path_to_target_segments(labels,scores,targets,0,duration,'drop')
     if ref_type=='common':peak_count+=len(tokens);peak_frames+=sum(t['end_frame']-t['start_frame'] for t in tokens)
     entry=dict(reference=ref,target_IPA=ipa,target_ids=targets,frame_token_ids=labels.tolist(),paths={})
     for policy in ['previous','drop']:
      segs=[dict(x) for x in tokens];h.apply_blank_policy(segs,n,policy)
      for clock in ['physical20ms','duration_per_frame']:
       step=.02 if clock=='physical20ms' else duration/n
       spans=[];cursor=0
       for phone,size in zip(phones,sizes):
        group=segs[cursor:cursor+size];cursor+=size;spans.append([phone,group[0]['start_frame']*step,min(duration,group[-1]['end_frame']*step)])
       # The physical clock fixes internal boundaries. Only the terminal utterance endpoint is closed at duration.
       if policy=='previous':spans[0][1]=0.;spans[-1][2]=duration
       assert all(0<=s<=e<=duration+1e-8 for _,s,e in spans)
       c=counters[ref_type,policy,clock];c['utterances']+=1
       funcs['score'](c,ref,[t for _,s,e in spans for t in(s,e)],duration)
       for (rp,rs,re),(pp,ps,pe) in zip(ref,spans):
        assert rp==pp;err=c['errors'];err['tokens']+=1;err['start']+=abs(rs-ps);err['end']+=abs(re-pe);err['start_hit20']+=abs(rs-ps)<=.02+1e-9;err['end_hit20']+=abs(re-pe)<=.02+1e-9
       entry['paths'][policy+'__'+clock]=spans
     if ref_type=='legacy':
      spans=entry['paths']['previous__duration_per_frame'];old=legacy[row['utt']]
      agree=max(abs(s-old['pred_starts'][j]) for j,(_,s,e) in enumerate(spans));agree=max(agree,max(abs(e-old['pred_ends'][j]) for j,(_,s,e) in enumerate(spans)));agreements.append(agree)
     record['alignments'][ref_type]=entry
    stream.write(json.dumps(record)+'\n')
    if (idx+1)%50==0 or idx+1==expected:stream.flush();print(split,idx+1,'/',expected,'seconds',round(time.monotonic()-started,1),flush=True)
  assert expected==len({json.loads(l)['utt'] for l in pred_file.open()})
  met={ '__'.join(k):funcs['summary'](v) for k,v in counters.items() }
  summary=dict(split=split,utterances=expected,modes=met,diagnostics=dict(legacy_path_replay_max_abs_difference_ms=1000*max(agreements),legacy_path_replay_mean_utterance_max_abs_difference_ms=1000*np.mean(agreements),greedy_blank_argmax_percent=100*blank_frames/total_frames,common_K2_nonblank_mean_run_frames=peak_frames/peak_count,common_K2_nonblank_mean_run_ms=20*peak_frames/peak_count,frames=total_frames),primary='common__previous__physical20ms')
  write(metrics_file,summary);write(HERE/f'{split}.COMPLETE.json',dict(status='complete',utterances=expected,predictions_sha256=sha(pred_file),metrics_sha256=sha(metrics_file),failed=0));all_metrics[split]=summary
 assert sha(CKPT)==provenance['checkpoint_sha256']
 write(HERE/'COMPLETE.json',dict(status='complete',dev=400,test=192,checkpoint_unchanged=True,seconds=time.monotonic()-started));print('ZIPA_K2_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
