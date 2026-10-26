"""Audit recognition completion and fill requested local table columns."""
from pathlib import Path
import csv,json,hashlib,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;O=ROOT/'reproduction/prior_models_results_20261022';K=ROOT/'reproduction/zipa_timit_k2_audit_20261022'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=load(HERE/'RESULTS.json');newrows=[];audits=[]
for model in ['zipa','powsm']:
 for split,n in [('dev',400),('test',192)]:
  p=HERE/model;d=load(p/f'{split}.json');c=load(p/f'{split}.COMPLETE.json');assert c['utterances']==n
  for f,key in [(f'{split}.json','metrics_sha256'),(f'{split}.predictions.jsonl','predictions_sha256'),(f'{split}.scores.jsonl','scores_sha256')]:assert sha(p/f)==c[key]
  rec=[json.loads(l) for l in (p/f'{split}.predictions.jsonl').open()];assert len(rec)==len({r['utt'] for r in rec})==n
  assert all(not r['reference_text_used'] for r in rec)
  scores=[json.loads(l) for l in (p/f'{split}.scores.jsonl').open()];assert len(scores)==n
  assert abs(100*sum(r['phone_errors'] for r in scores)/sum(r['reference_phones'] for r in scores)-d['PER'])<1e-9
  assert abs(100*sum(r['feature_edit_cost'] for r in scores)/sum(r['reference_IPA_segments'] for r in scores)-d['PFER'])<1e-9
  b=d['boundaries']['20'];newrows.append(dict(section='TIMIT纯音频识别补测',model='ZIPA Large CR-CTC-NS' if model=='zipa' else 'POWSM-CTC',dataset='TIMIT',split=split,n=n,mode='text_free',PER=d['PER'],PFER=d['PFER'],F1=b['F1'],R_value=b['R_value'],status='completed',protocol='shared ifmdd_arpabet; IPA→ARPAbet canonicalization; remove sil/q; dx→d; pooled PER/PanPhon vectors; physical20/40ms; no reference-conditioned decode',source=str(p/f'{split}.json')))
  audits.append(dict(model=model,split=split,n=n,unique_ids=n,hashes='matched',PER_PFER_counts='verified'))
L=['# ZIPA / POWSM-CTC：TIMIT 纯音频 PER/PFER','',
'完成日期：2026-10-22。各模型完整 dev400/test192，固定公开权重、FP32、batch1 greedy CTC；不输入参考文本、音素序列或参考数量，无微调。','',
'## 识别结果','',
'| 方法 | split | 句数 | PER | PFER | 无文本 F1@20ms | 无文本 R-value |','|---|---|---:|---:|---:|---:|---:|']
for r in newrows:L.append(f"| {r['model']} | {r['split']} | {r['n']} | {r['PER']:.2f} | {r['PFER']:.2f} | {r['F1']:.2f} | {r['R_value']:.2f} |")
L+=['',
'单位均为%。这里 PER 是音频解码假设的音素编辑率；与 POWSM 旧文字输入 FA 报告中的 G2P PER 不同。','',
'## 共同比较协议','',
'两模型均用与当前 OTTC sweep 相同的 TIMIT 清单及 `ifmdd_arpabet` 参考；评分前去sil/q、dx→d，语料级累计编辑成本/累计参考长度，不按句平均。IPA输出经两者共享的冻结 ZIPA reverse map 转ARPAbet；另显式将ɾ→d、ʔ→q（随后去q）、ɡ→g并合并带tie的affricate。映射和丢弃的modifier计数记录在JSON；该结果是映射后的TIMIT协议分数，不能当作论文原始IPA PER/PFER。','',
'PFER使用当前OTTC/HuPER表相同的冻结24维PanPhon0.22.2特征表、双元音展开方式和feature_distance后端。对于无ARPAbet对应的有效IPA，保留独立PER token并使用其PanPhon特征；残余rhotic modifier附到前一个token。对于没有任何PanPhon向量的符号（本次POWSM有R/Q），不静默删除：作为独立PER token，并将PFER插入/删除/替换代价设为完整单位1；其他有效向量的原成本不变。该明确标注的未知符号扩展不同于直接让PanPhon丢弃未知字符。','',
'无文本分割由同一份audio-only输出产生，ZIPA物理20ms、POWSM物理40ms；合并IPA到同一ARPAbet occurrence后取下一音素起点，参考唯一端点、去句首尾20ms、chronological one-to-one、冻结IF-MDD R-value。与历史ZIPA不同时间轴/不同边界参考的50.05/55.23不混报。','',
'## 模型与覆盖证据','',
'- ZIPA：固定Large CR-CTC-NS800k Avg10，权重SHA256 `dcf8d4b507d0eacbac40735dd742a4c85071c52aab35255cc27da644f68b8584`。','- POWSM-CTC：固定官方70epoch checkpoint，权重SHA256 `d9d56a166e5dfe3088ca2080c8c3e2eb8124827166a68cd41b673963f99c9ef8`。使用<unk>/<pr>/<na>前缀，官方20s零填充buffer，移除两个prefix encoder位置，保留真实音频帧，无截断。','- 各模型592句推理完整，dev/test的ID唯一；predictions/scores/metrics hash一致，PER/PFER从逐句成本重新汇总验证；权重hash前后未变。','- GPU作业3674213.opbs已完成两模型全部推理；首次评分因不可映射符号停止，因此该作业退出码1，不宣称整个作业退出0。保留原job.log/pbs_final.txt。修正明确的未知符号规则后，直接复用完整预测，本地CPU评分退出0，四组COMPLETE及总COMPLETE均存在，score.log以TIMIT_IPA_RECOGNITION_COMPLETE结束。','- `provenance.json`记录模型与清单hash，`RESULTS.json`记录评分源/映射hash；各模型目录保留每split预测、逐句评分与完成标记。本次所有结果仅本地保存。','',
'## 主表更新规则','',
'主表ZIPA行现在使用本次audio-only PER/PFER及无文本分割，并配套已完成的新K2 FA列（sil/q从FA目标中排除，协议差异仍标注）；旧ZIPA历史行保留在历史报告。POWSM行识别/无文本列使用本次纯音频结果，FA列仍为历史word-transcript FA-Bench，保持其输入条件标记；未把FA输出当识别假设。','']
(HERE/'REPORT_ZH.md').write_text('\n'.join(L)+'\n')
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
write(HERE/'AUDIT.json',dict(status='passed',sets=audits,local_scoring_exit=0,pbs_initial_scoring_exit=1))
# Primary table: only one standard CIF baseline, no disappearance of the new K2 result.
all_tables={}
for split in ['dev','test']:
 p=O/f'TIMIT_{split.upper()}_TABLE.csv'
 with p.open(encoding='utf-8-sig') as f:reader=csv.DictReader(f);cols=reader.fieldnames;rows=list(reader)
 rows=[r for r in rows if not r['方法'].startswith('ZIPA K2重算')]
 for r in rows:
  if r['方法'].startswith('ZIPA Large'):
   x=results['results']['zipa'][split];fa=load(K/f'{split}.json')['modes']['common__previous__physical20ms'];b=x['boundaries']['20'];r.update({'方法':'ZIPA Large CR-CTC-NS‡','PER':f"{x['PER']:.2f}",'PFER':f"{x['PFER']:.2f}",'无文本 F1':f"{b['F1']:.2f}",'无文本 R-value':f"{b['R_value']:.2f}",'FA F1':f"{fa['boundaries']['20']['f1_percent']:.2f}",'FA R-value':f"{fa['boundaries']['20']['r_value_percent']:.2f}",'MAE (ms)':f"{fa['boundary_MAE_ms']:.2f}"})
  if r['方法'].startswith('POWSM'):
   x=results['results']['powsm'][split];b=x['boundaries']['20'];r.update({'方法':'POWSM-CTC§','PER':f"{x['PER']:.2f}",'PFER':f"{x['PFER']:.2f}",'无文本 F1':f"{b['F1']:.2f}",'无文本 R-value':f"{b['R_value']:.2f}"})
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 all_tables[split]=(cols,rows)
p=O/'TIMIT_TABLES_ZH.md';s=p.read_text()
for split,marker in [('test','## Test（192句）'),('dev','## Dev（400句）')]:
 cols,rows=all_tables[split];start=s.index('| 方法',s.index(marker));end=s.index('\n\n',start)
 table='\n'.join(['| '+' | '.join(cols)+' |','|---|'+'|'.join(['---:']*7)+'|']+['| '+' | '.join(row[c] for c in cols)+' |' for row in rows]);s=s[:start]+table+s[end:]
s=s.replace('‡ ZIPA 使用历史 silence-filtered ARPAbet 内部边界协议。此行 MAE=FA 全音素起止误差平均（test38.75ms），不是 text-free matched MAE@20ms=9.90ms。PER/PFER 本次未统一重评。','‡ ZIPA 已补全本次纯音频PER/PFER及无文本分割；FA列使用新K2/previous blank/物理20ms，排除sil/q目标并计算保留音素MAE。历史38.75ms及旧无文本50.05/55.23另留历史报告。')
s=s.replace('§ POWSM-CTC 的 test 行为 clean、参考文字→Charsiu G2P→CTC FA；F1 为 FA-Bench context/time 匹配，MAE 为 manner matching。其原报告 PER 是 G2P 编辑率，因此不填入 audio-only PER 列。该评测未提供 R-value；dev 未评测。','§ POWSM-CTC 的PER/PFER/无文本列来自新audio-only评测；FA列为历史clean、参考文字→Charsiu G2P→CTC FA，F1用FA-Bench context/time、MAE用manner matching；未提供FA R-value，dev FA未评测。旧G2P PER未填入识别列。')
s+=f'\n## ZIPA / POWSM 识别补齐\n\n[完整报告与映射规则]({HERE}/REPORT_ZH.md)。两者dev/test均已完整评分；主表PER为共同ARPAbet协议，PFER为共同特征表上的映射后分数，不是论文原始IPA分数。不可映射POWSM符号保留单位编辑代价，未静默删除。\n'
p.write_text(s)
p=O/'RESULTS.json';d=load(p);d['records']=[r for r in d['records'] if r.get('section')!='TIMIT纯音频识别补测']+newrows;d['timit_ipa_recognition']=str(HERE/'REPORT_ZH.md');write(p,d);fields=list(dict.fromkeys(k for r in d['records'] for k in r))
with (O/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(d['records'])
p=O/'REPORT_ZH.md';s=p.read_text();s+=f'\n## ZIPA / POWSM TIMIT audio-only识别已补齐\n\n[报告]({HERE}/REPORT_ZH.md)。test192：ZIPA PER/PFER26.45/11.52，POWSM42.20/14.08。dev400分别23.47/10.26和39.55/13.18。已更新指定八列表头主表；这些是共同ARPAbet映射后识别分数，不是word-G2P或FA输出编辑率。\n';p.write_text(s)
print('AUDIT PASSED; PER/PFER filled in both dev/test primary tables.')
