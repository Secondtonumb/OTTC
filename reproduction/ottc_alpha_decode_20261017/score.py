"""Score completed audio-only decoding; references enter only this script."""
import ast,csv,hashlib,json,os
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def edit_distance(a,b):
 previous=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  current=[i]
  for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
  previous=current
 return previous[-1]


def main():
 from reproduction.segmentation_metrics import _scorer,SOURCE
 complete=json.loads((HERE/'DECODE_COMPLETE.json').read_text());assert complete['utterances']==4
 refs=json.loads((HERE/'SCORING_REFERENCES.json').read_text())
 cases=[json.loads(s) for s in (HERE/'predictions.jsonl').read_text().splitlines()]
 matcher=ROOT/'reproduction/buckeye_singlehead_20261014/train.py'
 fn=next(n for n in ast.parse(matcher.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='boundary_matches')
 ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(matcher),'exec'),ns);match=ns['boundary_matches']
 totals={};rows=[];summaries=[]
 normalize=lambda xs:['d' if p=='dx' else p for p in xs if p not in ('sil','q')]
 for case in cases:
  rid=case['recording_id'];reference=refs[rid]['reference'];refphones=normalize(reference['phones']);hyp=normalize(case['hypothesis'])
  errors=edit_distance(refphones,hyp);case['normalized_nosil_PER_percent']=100*errors/len(refphones)
  metrics={}
  for mode,key in [('argmax','argmax_boundaries'),('alpha_grid','native_alpha_grid_boundaries'),('alpha_fractional','native_alpha_boundaries')]:
   bounds=sorted(set(float(t) for t in case[key] if .02<t<case['duration']-.02))
   metrics[mode]={}
   for ms in [10,20,25]:
    hit=match(reference['boundaries'],bounds,ms/1000)
    metric=_scorer.finalize_counts(hit,hit,len(bounds),len(reference['boundaries']))
    metrics[mode][str(ms)]=metric
    rows.append(dict(recording_id=rid,dataset=case['dataset'],method=mode,tolerance_ms=ms,**{k:metric[k] for k in ['precision_pct','recall_pct','f1_pct','r_value_pct']}))
    counter=totals.setdefault(case['dataset'],{}).setdefault(mode,{}).setdefault(str(ms),Counter())
    counter.update(matches=hit,predicted=len(bounds),reference=len(reference['boundaries']))
  case['metrics']=metrics
  case['rounded_boundary_collisions']=len(case['native_alpha_grid_boundaries'])-len(set(case['native_alpha_grid_boundaries']))
  summaries.append(case)
 (HERE/'scored_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in summaries))
 jobs=[]
 for model in complete['models']:
  dataset=model['dataset'];model_cases=[c for c in cases if c['dataset']==dataset]
  job=dict(model,scope='selected visualization examples only; not full benchmark',
   methods={method:{ms:_scorer.finalize_counts(c['matches'],c['matches'],c['predicted'],c['reference']) for ms,c in by_ms.items()} for method,by_ms in totals[dataset].items()},
   recognition_sequence_unchanged=True,optimizer_updates=0)
  jobs.append(job)
 result=dict(scope='four existing examples, three single-head upstream-JSD checkpoints',jobs=jobs,
  reference_used_only_by_scorer=True,reference_quality='TIMIT/Buckeye human phones; Libri automatic aligner identity unverified',
  boundary_protocol='same fixed20ms frame-cell clock for all three methods; unique edges; exclude20ms utterance edges; one-to-one 10/20/25ms',
  beta='1/predicted hypothesis length, not reference phone count',
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),SOURCE,matcher,HERE/'SCORING_REFERENCES.json',HERE/'DECODE_COMPLETE.json']})
 (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 with (HERE/'BOUNDARY_METRICS.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 def cell(c,mode):
  m=c['metrics'][mode]['20'];return f'{m["f1_pct"]:.2f} / {m["r_value_pct"]:.2f}'
 lines=['# 单头 OTTC upstream-JSD：无文本 α 质量解码','',
 '已实际推理三个既有模型，在四条既定可视化示例上比较argmax边界与α质量解码；未重训、未用参考音素数/序列/时间解码。仅四条示例，不是全测试集。','',
 '识别序列与之前保存结果逐条一致。使用自身识别序列长度Û，设置每个音素目标质量1/Û，对α累计质量做逆CDF切分；跨frame边界按帧内均匀质量假设插值。保存的运输矩阵已检查行和=α、列和=1/Û。','',
 '两项变化分别比较：argmax→α重分段可能移动整句边界；α取整→α分数帧单独衡量插值作用。三种结果使用完全相同的固定20ms网格，无拟合时间偏移。新增分段不改变识别音素/PER。','',
 '**下表为所选整句 F1 / R-value（%，20ms容差）**。','',
 '|语音|原argmax|同一α取整到20ms|α分数帧|预测音素数|平均帧内位移ms|','|---|---:|---:|---:|---:|---:|']
 for c in cases:lines.append(f'|{c["recording_id"]}|{cell(c,"argmax")}|{cell(c,"alpha_grid")}|{cell(c,"alpha_fractional")}|{c["predicted_phone_count"]}|{c["mean_abs_subframe_shift_ms"]:.2f}|')
 lines+=['','10/20/25ms的P/R/F1/R-value见BOUNDARY_METRICS.csv。参考仅由本评分脚本读取；GPU解码输入与评分参考分文件保存。所有原checkpoint严格加载、hash与先前评估一致、推理前后未改动。',
 '', '更细的时间戳是基于均匀帧内质量的估计，不保证比20ms网格更准确。音素数预测错误会改变切分；原识别序列缺少sil时，本方法也不会额外生成sil/静音路径。blank不被自动视为静音。<tail>只补齐最后不足一个frame的显示时间，不新增评分点。',
 '', '复现：在PBS分配的GH200上执行 `bash reproduction/ottc_alpha_decode_20261017/run.sh`，随后CPU运行 `python -m reproduction.ottc_alpha_decode_20261017.score`。',
 '', '[打开Recognition子页](../segmentation_figures_20261017/interactive/recognition.html)，勾选OTTC upstream-JSD (α)与原OTTC upstream-JSD比较。']
 (HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines[10:17]))
 # Inspectable native alpha and inverse-CDF curves, independent of reference labels.
 os.environ.setdefault('MPLCONFIGDIR',str(HERE/'matplotlib_cache'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
 for c in cases:
  mass=np.asarray(c['alpha']);edge=np.arange(len(mass)+1)*.02;cdf=np.r_[0,mass.cumsum()]
  fig,axes=plt.subplots(2,1,figsize=(14,5),sharex=True)
  axes[0].bar(edge[:-1],mass,width=.02,align='edge',color='#006d77');axes[0].set_ylabel('Predicted frame mass α')
  axes[1].plot(edge,cdf,color='#006d77');axes[1].set_ylabel('Cumulative mass')
  for j,t in enumerate(c['native_alpha_boundaries'],1):
   q=j/c['predicted_phone_count'];axes[1].plot(t,q,'o',color='#c84b31',markersize=3)
   axes[1].axvline(t,color='#c84b31',alpha=.15,linewidth=.7)
  axes[1].set_xlim(.5,min(1.8,c['duration']));axes[1].set_xlabel('Time (s); points are predicted-sequence quantile crossings, not reference boundaries')
  axes[0].set_title(c['recording_id']+' — learned α and fractional boundaries')
  fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(figdir/(c['recording_id']+'_alpha_zoom.'+ext),dpi=160)
  plt.close(fig)
 (HERE/'SCORE_COMPLETE.json').write_text(json.dumps(dict(status='complete',utterances=4,methods=['argmax','alpha_grid','alpha_fractional'],reference_used_only_by_scorer=True),indent=2)+'\n')

if __name__=='__main__':main()
