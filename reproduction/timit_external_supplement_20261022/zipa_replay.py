"""Replay saved ZIPA paths; supplemental conditional boundary error, not FA MAE."""
import csv, hashlib, importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SCRIPTS=Path('/work/gm64/m64000/IF-MDD/fa_research/scripts');sys.path.insert(0,str(SCRIPTS))
from boundary_rvalue import score_events, finalize_counts
from score_saved_inference_segmentation_jsonl import boundary_events
BASE=Path('/work/gm64/m64000/IF-MDD/fa_research/results/zipa_large_crctc_ns_800k_cmu_monophone_20260624/zipa_large_crctc_ns_800k')
HIST=Path('/work/gm64/m64000/zipa/fa_research/results/zipa_crctc_ns_timit_arpabet_rvalue_20260810')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pairs(ref,queries,tol=.02):
 used=set();out=[]
 for qi,(qt,_) in enumerate(queries):
  for ri,(rt,_) in enumerate(ref):
   if ri not in used and abs(rt-qt)<=tol:
    used.add(ri);out.append((qi,ri,abs(rt-qt)));break
 return out
result={};files={};out=HERE/'zipa';out.mkdir(exist_ok=True)
for split,expected in [('timit_dev',400),('timit_test',192)]:
 f=BASE/split/'text_free/alignments.jsonl';rows=[json.loads(l) for l in f.open() if l.strip()];assert len(rows)==expected and len({r['utt'] for r in rows})==expected
 pm=rm=np=nr=0;pe=re=pn=rn=0.;utterances=[]
 for r in rows:
  for pre in ['pred','gt']:
   phones=r['phones' if pre=='pred' else 'gt_phones'];assert len(phones)==len(r[pre+'_starts'])==len(r[pre+'_ends'])
  pred=boundary_events(r['phones'],r['pred_starts'],r['pred_ends'],filter_silence=True);ref=boundary_events(r['gt_phones'],r['gt_starts'],r['gt_ends'],filter_silence=True)
  a=pairs(ref,pred);b=pairs(pred,ref);met=score_events(pred,ref,.02)
  assert len(a)==met['precision_matches'] and len(b)==met['recall_matches'];pm+=len(a);rm+=len(b);np+=len(pred);nr+=len(ref)
  ep=sum(x[2] for x in a);er=sum(x[2] for x in b);pe+=ep;re+=er
  nearp=sum(min(abs(pt-rt) for rt,_ in ref) for pt,_ in pred) if ref else 0
  nearr=sum(min(abs(rt-pt) for pt,_ in pred) for rt,_ in ref) if pred else 0
  assert pred and ref;pn+=nearp;rn+=nearr
  utterances.append(dict(utt=r['utt'],predicted=len(pred),reference=len(ref),precision_matches=len(a),recall_matches=len(b),precision_matched_abs_error_s=ep,recall_matched_abs_error_s=er,pred_to_ref_nearest_error_s=nearp,ref_to_pred_nearest_error_s=nearr))
 met=finalize_counts(pm,rm,np,nr);history=HIST/split/'text_free/tol20ms/inference_segmentation_prf_rvalue_summary.json';old=json.loads(history.read_text())['groups'][0]
 for key in ['precision_pct','recall_pct','f1_pct','r_value_pct']:assert abs(met[key]-old['boundary_only_'+key])<1e-8
 result[split]=dict(utterances=expected,failures=0,boundary_20ms=met,matched_boundary_MAE_20ms=1000*(pe+re)/(pm+rm),precision_matched_MAE_20ms=1000*pe/pm,recall_matched_MAE_20ms=1000*re/rm,pred_to_ref_nearest_MAE_ms=1000*pn/np,ref_to_pred_nearest_MAE_ms=1000*rn/nr,symmetric_nearest_MAE_ms=500*(pn/np+rn/nr),matches_MAE_denominator=pm+rm,original_metric_reproduced=True)
 with (out/f'{split}.per_utterance.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(utterances[0]));w.writeheader();w.writerows(utterances)
 files[str(f)]=digest(f);files[str(history)]=digest(history)
for f in [Path(__file__),SCRIPTS/'boundary_rvalue.py',SCRIPTS/'score_saved_inference_segmentation_jsonl.py']:files[str(f)]=digest(f)
res=dict(status='complete',type='historical_prediction_replay_backfill',new_inference=False,new_training=False,evaluation_uploads=False,protocol={'boundary':'exact historical silence-filtered internal ends; directional one-to-one IF-MDD matching at20ms','matched_MAE':'pooled absolute distance over both directional matched pair lists, conditional on match within20ms; excludes misses; not all-phone FA MAE','nearest_MAE':'all internal boundaries; nearest in other set, separately per direction; symmetric average of directional means; no tolerance or one-to-one restriction','time_clock':'unchanged saved historical duration/frame timestamps; no retiming','primary_metrics':'P/R/F1/R-value unchanged'},splits=result,source_sha256=files)
(out/'RESULTS.json').write_text(json.dumps(res,indent=2)+'\n');(out/'COMPLETE.json').write_text(json.dumps(dict(status='complete',dev=400,test=192,results_sha256=digest(out/'RESULTS.json')),indent=2)+'\n')
print(json.dumps(result,indent=2))
