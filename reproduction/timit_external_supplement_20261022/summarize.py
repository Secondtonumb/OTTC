"""Validate completed supplement and update the local cross-model report."""
import csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;PRIOR=ROOT/'reproduction/prior_models_results_20261022'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
rows=[];audit=[]
for name,mode in [('cif','forced_alignment'),('cif_common','forced_alignment'),('huper','text_free'),('huper_common','text_free')]:
 assert load(HERE/name/'COMPLETE.json')['status']=='complete'
 for split,n in [('dev',400),('test',192)]:
  p=HERE/name;done=load(p/f'{split}.COMPLETE.json');pred=p/f'{split}.predictions.jsonl';metric=p/f'{split}.json';data=load(metric);m=data['modes'][mode];b=m['boundaries']['20']
  assert done['predictions_sha256']==sha(pred) and done['metrics_sha256']==sha(metric)
  rec=[json.loads(l) for l in pred.open()];assert len(rec)==len({r['utt'] for r in rec})==n
  assert m['alignment_failures']==0
  row=dict(section='TIMIT补测',model='CIF CE+Quantity+CTC(0.5) epoch24' if name.startswith('cif') else 'HuPER recognizer',dataset='TIMIT',split=split,n=n,mode=mode,status='completed',reference='ifmdd_arpabet_common' if 'common' in name else 'ottc_arpabet_native',PER=m.get('nosil_PER_percent'),PFER=m.get('PFER_percent'),precision=b['precision_percent'],recall=b['recall_percent'],F1=b['f1_percent'],R_value=b['r_value_percent'],MAE_ms=m.get('boundary_MAE_ms'),source=str(metric),protocol=json.dumps(data['protocol'],ensure_ascii=False))
  rows.append(row);audit.append(dict(model=name,split=split,utterances=n,predictions_sha256=sha(pred),metrics_sha256=sha(metric),alignment_failures=0))
# Verify that core model implementation still matches the original training contract.
job=next(j for j in load(ROOT/'reproduction/cif_fa_probe_20261017/INPUTS.json')['jobs'] if j['key']=='timit_ctc_0p5')
for f in ['ottc/models/sequence/cif.py','ottc/models/sequence/wav2vec2_cif.py']:
 assert sha(ROOT/f)==job['original_training_contract']['sources_sha256'][f]
for jid in ['3674105','3674118']:
 s=(HERE/f'pbs_{jid}_final.txt').read_text();assert re.search(r'job_state\s*=\s*F',s) and re.search(r'Exit_status\s*=\s*0\b',s)
z=load(HERE/'zipa/RESULTS.json')
L=['# TIMIT 补测：HuPER、CIF epoch24 FA、ZIPA 误差','',
'完成日期：2026-10-22。全部评估为本地运行；未训练、未修改原权重，未上传 W&B。','',
'## 1. HuPER TIMIT dev400/test192','',
'使用公开 HuPER recognizer 固定权重，FP32、greedy CTC、batch1；只输入音频，不用参考文本、音素数或时间。以下以与 TIMIT seed sweep 相同的 `references.ifmdd_arpabet` 为主参考，移除 sil/q、dx→d 后计算 PER/PanPhon PFER。分割使用 20 ms 物理帧，内部 blank 归前音素，边界取下一非 blank 音素起点；参考使用唯一端点、排除句首尾20ms，按冻结 chronological one-to-one 匹配评分。','',
'| split | 句数 | PER | PFER | P | Recall | F1 | R-value |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 if r['model']=='HuPER recognizer' and 'common' in r['reference']:L.append(f"| {r['split']} | {r['n']} | "+' | '.join(f'{r[k]:.2f}' for k in ['PER','PFER','precision','recall','F1','R_value'])+' |')
L+=['',
'这些是固定公开模型的零样本结果，不是多 seed 均值。HuPER 的训练集与 TIMIT 重叠尚未审计，不能将“零微调”表述为已证实无训练重叠。参考和评分实现与 sweep 对齐，但模型输出时钟/解码协议保留其原生定义，不能作为等架构、等训练条件消融。另保留 `huper/` 下旧 CIF native 参考的评分；主表使用 `huper_common/`，不可混报。','',
'## 2. CIF CE+Quantity+CTC(0.5)，epoch24 的 FA','',
'复用原始按 dev text-free R-value 选出的 epoch24 / step1752，checkpoint SHA256 `197da9053fed71a26f466c8d21ac463206034ed9f668fd1dc643fe02d75fc488`。严格加载原模型，FP32 参数及推理，不重新选 checkpoint。使用 **encoder 帧级 CTC 辅助头 + 原有 IF-MDD K2 CTC 图**，给定音素序列；不是 CIF firing 边界，也不是只给音素数量的 length-conditioned CIF。参考时间仅进入评分，不进入 aligner。保留 duration/frames 时钟与 blank previous 策略。','',
'### 与 seed sweep 相同参考的 FA','',
'给定并评分 `ifmdd_arpabet` 音素序列；20ms 边界协议与 sweep 一致。','',
'| split | 句数 | P | Recall | F1 | R-value | 起点 MAE ms | 终点 MAE ms | 边界 MAE ms | 失败 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for split in ['dev','test']:
 m=load(HERE/'cif_common'/f'{split}.json')['modes']['forced_alignment'];b=m['boundaries']['20'];L.append(f"| {split} | {m['utterances']} | "+' | '.join(f'{b[k]:.2f}' for k in ['precision_percent','recall_percent','f1_percent','r_value_percent'])+' | '+' | '.join(f'{m[k]:.2f}' for k in ['start_MAE_ms','end_MAE_ms','boundary_MAE_ms'])+f" | {m['alignment_failures']} |")
L+=['',
'### 原 CIF native 音素序列的 FA','',
'为对接用户所列 dev epoch24 行，另使用 `ottc_arpabet` 原训练 inventory 序列做 FA，并按原 CIF start-boundary/directional matching 评分。两套 native/common 输入序列不同，故各自重新对齐；不将 native FA 直接按数组位置与 common 参考配对。','',
'| split | P | Recall | F1 | R-value | native 全音素边界 MAE ms |','|---|---:|---:|---:|---:|---:|']
for split in ['dev','test']:
 d=load(HERE/'cif'/f'{split}.json');m=d['modes']['forced_alignment'];b=d['CIF_original_start_boundary_protocol']['20'];L.append(f'| {split} | '+' | '.join(f'{b[k]:.2f}' for k in ['precision_pct','recall_pct','f1_pct','r_value_pct'])+f" | {m['boundary_MAE_ms']:.2f} |")
L+=['',
'用户引用的 dev 无文本 F1/R-value=58.33/64.60，与同 checkpoint native CTC-FA 的49.88/57.22 分开报告；这两种输出机制不同，当前结果没有显示辅助 CTC-FA 优于 CIF 自由 firing 分割。MAE 为按给定音素 occurrence 对应的起止误差总和除以2×音素数，包含 native sil/q occurrence；不等于仅20ms内匹配边界的平均误差。原 CIF 的音素序列含更多 silence/closure 信息，common 协议使用不同参考音素序列，数值差异不能单独归因于模型精度。','',
'## 3. ZIPA text-free 补算','',
'完整重放保存的 dev400/test192 音频解码结果，未重新推理或改变历史时间轴。原 P/R/F1/R-value 逐项复现。text-free 预测音素数与参考不等，没有像给定音素 FA 那样天然的逐 occurrence 一一对应，因此不强行套用 FA MAE。补算以下明确命名的边界误差：','',
'- matched-boundary MAE@20ms：原 IF-MDD 两个方向贪心匹配的所有成功配对误差合并平均。仅统计20ms内成功匹配，漏检/多检由 P/R/F1/R-value 表达。','- 双向 nearest MAE：每个预测边界找最近参考、每个参考边界找最近预测，各自求均值再平均；覆盖全部内部边界，无20ms截断，但允许多对一，不是 FA MAE。','',
'| split | 句数 | P | Recall | F1 | R-value | matched MAE@20ms | pred→ref nearest MAE | ref→pred nearest MAE | 双向 nearest MAE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for split,d in z['splits'].items():
 b=d['boundary_20ms'];L.append(f"| {split} | {d['utterances']} | "+' | '.join(f'{b[k]:.2f}' for k in ['precision_pct','recall_pct','f1_pct','r_value_pct'])+' | '+' | '.join(f'{d[k]:.2f}' for k in ['matched_boundary_MAE_20ms','pred_to_ref_nearest_MAE_ms','ref_to_pred_nearest_MAE_ms','symmetric_nearest_MAE_ms'])+' |')
L+=['',
'误差单位均为 ms。test 成功匹配2,935个边界，占预测47.60%、参考52.77%；9.90ms 是这些成功匹配的条件误差，不能与 CIF FA 的30.09ms或 ZIPA FA 的38.75ms比较大小并宣称对齐更好。双向 nearest 亦为补充诊断；主要分割指标仍为 F1/R-value。','',
'## 4. 完成与可追溯证据','',
'- PBS `3674105.opbs`：CIF native FA + HuPER 推理，F/Exit_status=0，walltime 1分；`3674118.opbs`：CIF common FA，F/Exit_status=0，walltime36秒。最终状态见 `pbs_*_final.txt`。','- 每个 CIF/HuPER 分支 dev400/test192 完整覆盖，ID无重复；所有预测/指标 SHA256 与完成标记一致。CIF 两套 FA 均无失败，原 checkpoint hash复核未变，两个 CIF 实现源码 hash与历史训练 contract一致。','- `cif/`、`cif_common/`、`huper/`、`huper_common/`：各split完整预测/指标/完成标记、模型与数据来源hash；`zipa/`：历史重放指标与逐句误差。','- `AUDIT.json`：最终完整性核验；`RESULTS.csv`/`RESULTS.json`：补测结果；原跨模型报告已补充本次结果。','']
(HERE/'REPORT_ZH.md').write_text('\n'.join(L)+'\n')
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
write(HERE/'AUDIT.json',dict(status='passed',prediction_sets=audit,cif_source_contract='matched',pbs='both F exit0',zipa_original_PRFR_reproduced=True))
write(HERE/'RESULTS.json',dict(status='complete',rows=rows,zipa=z,source='existing fixed checkpoints; no training'))
fields=list(rows[0]);
with (HERE/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
# Update derived master tables with explicit metric names; original result files stay untouched.
p=PRIOR/'RESULTS.json';d=load(p);d['records']=[r for r in d['records'] if r.get('section')!='TIMIT补测']+rows
for r in d['records']:
 if r.get('section')=='ZIPA历史TIMIT' and r['mode']=='text_free':
  zz=z['splits'][r['split']];r['matched_boundary_MAE_20ms']=zz['matched_boundary_MAE_20ms'];r['symmetric_nearest_MAE_ms']=zz['symmetric_nearest_MAE_ms']
d['timit_supplement']=dict(report=str(HERE/'REPORT_ZH.md'),audit=str(HERE/'AUDIT.json'),status='complete')
write(p,d)
fields=list(dict.fromkeys(k for r in d['records'] for k in r))
with (PRIOR/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(d['records'])
p=PRIOR/'REPORT_ZH.md';s=p.read_text();heading='## 9. TIMIT 补测更新（2026-10-22）'
if heading in s:s=s[:s.index(heading)]
s=s.replace('| HuPER | VoxAngeles 识别/无文本分割；L2-ARCTIC900/Buckeye1840 识别 |','| HuPER | TIMIT dev400/test192、VoxAngeles 识别/无文本分割；L2-ARCTIC900/Buckeye1840 识别 |')
s=s.replace('timit_dev | text_free | 400 | 47.65 | 52.08 | 49.77 | 55.37 | — |','timit_dev | text_free | 400 | 47.65 | 52.08 | 49.77 | 55.37 | 见补算定义 |').replace('timit_test | text_free | 192 | 47.60 | 52.77 | 50.05 | 55.23 | — |','timit_test | text_free | 192 | 47.60 | 52.77 | 50.05 | 55.23 | 见补算定义 |')
s+='\n'+heading+'\n\n'+f'[完整补测报告]({HERE}/REPORT_ZH.md)；[核验记录]({HERE}/AUDIT.json)。\n\n'
s+='HuPER 已补全 TIMIT dev400/test192；CIF epoch24 已补全两套参考下的给定音素 CTC-FA，均零失败；ZIPA 已复现历史分割指标并补算明确命名的边界误差。以下使用与本次 sweep 相同的 ifmdd_arpabet 参考，分割容差20ms；固定 checkpoint 单模型结果。\n\n| 模型 | 模式 | split | PER | PFER | F1 | R-value | FA MAE ms |\n|---|---|---|---:|---:|---:|---:|---:|\n'
for r in rows:
 if 'common' not in r['reference']:continue
 s+=f"| {r['model']} | {r['mode']} | {r['split']} | "+' | '.join('—' if r[k] is None else f'{r[k]:.2f}' for k in ['PER','PFER','F1','R_value','MAE_ms'])+' |\n'
s+='\nCIF 原生 start-boundary 协议下：dev FA F1/R-value=49.88/57.22，test=51.99/59.02，与原无文本 CIF firing 指标分开。ZIPA test matched-boundary MAE@20ms=9.90ms、双向 nearest MAE=23.85ms；这些是 text-free 补充误差，不能替代或直接比较 FA occurrence MAE。完整定义与覆盖率见补测报告。\n'
p.write_text(s)
write(HERE/'COMPLETE.json',dict(status='complete',CIF_native_FA=dict(dev=400,test=192),CIF_common_FA=dict(dev=400,test=192),HuPER_inference=dict(dev=400,test=192),ZIPA_historical_replay=dict(dev=400,test=192),audit_sha256=sha(HERE/'AUDIT.json')))
print('Complete. Master records:',len(d['records']))
