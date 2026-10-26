"""Close the high-PER audit with explicit language conditions and fresh evidence."""
import csv,json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;O=ROOT/'reproduction/prior_models_results_20261022'
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
eng=load(HERE/'RESULTS_ENG.json')['results']['powsm_eng'];old=load(HERE/'RESULTS.json')['results']['powsm'];z=load(HERE/'RESULTS.json')['results']['zipa'];probe=load(HERE/'audit/DECODER_PROBE_COMPLETE.json')
assert probe['right20_exact_matches']==probe['utterances']==18
for split,n in [('dev',400),('test',192)]:
 p=HERE/'powsm_eng';c=load(p/f'{split}.COMPLETE.json');assert c['utterances']==n
 for f,key in [(f'{split}.json','metrics_sha256'),(f'{split}.predictions.jsonl','predictions_sha256'),(f'{split}.scores.jsonl','scores_sha256')]:assert sha(p/f)==c[key]
 rows=[json.loads(l) for l in (p/f'{split}.predictions.jsonl').open()];assert len(rows)==len({r['utt'] for r in rows})==n
for p in [HERE/'audit/pbs_decoder_final.txt',HERE/'audit/pbs_eng_final.txt']:
 s=p.read_text();assert re.search(r'job_state\s*=\s*F',s) and re.search(r'Exit_status\s*=\s*0\b',s)
L=['# 高 PER 复核：POWSM 语言条件与 ZIPA 评分','',
'结论：POWSM 的高错误率有实质性推理条件因素。相同checkpoint、音频、FP32、CTC解码、评分器，只将<unk>语言前缀改为已知英语的<eng>，完整test PER由42.20%降至27.23%。这不是单纯修改映射后得到的改善。ZIPA26.45%与历史26.54%接近，未发现同类前缀问题。','',
'## 1. 完整数据对照','',
'| 模型/语言条件 | split | 句数 | PER | PFER | 无文本F1@20ms | 无文本R-value |','|---|---|---:|---:|---:|---:|---:|']
for name,result in [('ZIPA',z),('POWSM <unk>',old),('POWSM <eng>',eng)]:
 for split,n in [('dev',400),('test',192)]:
  x=result[split];b=x['boundaries']['20'];L.append(f"| {name} | {split} | {n} | {x['PER']:.2f} | {x['PFER']:.2f} | {b['F1']:.2f} | {b['R_value']:.2f} |")
L+=['',
'英语条件来自TIMIT语料语言元数据，不输入转写、目标音素或时间。先在18个dev诊断句上发现差异，再运行完整dev400/test192；未在test上搜索语言标签、阈值或映射。两个条件原文件都保留。主表明确使用POWSM<eng>，<unk>作为对照，不将其描述为无效运行或篡改旧值。','',
'## 2. 独立解码核验','',
'在12个按dev顺序均匀取样的句子及6个高错误dev句上，用官方安装版Speech2TextGreedySearch._decode_single_sample核对；相同right20条件的18/18条acoustic frame path逐帧复现旧输出。排除了此次抽样中的随机波动或错误保存路径。还检查left2s/left4s padding与<eng>前缀，保留全部诊断预测；它们不是完整test成绩。','',
'原本本地decode_powsm_shar.py声称<eng>在<pr>任务会几乎全blank，本次同权重实测不支持该说明；不再将该历史说明当作TIMIT的既定约束。模型卡本身的示例确实使用<unk>，因此<unk>是一个有效默认条件，但不能据此推断已知英语条件与默认条件性能相同。','',
'官方来源：[POWSM-CTC模型卡](https://huggingface.co/espnet/powsm_ctc)（16kHz、20秒输入、<unk>/<pr>示例及phone slash token说明）；本次实际执行源码hash保存在audit/DECODER_PROBE_COMPLETE.json。','',
'## 3. 映射敏感性','',
'相同旧假设，仅把ao→aa、zh→sh归并时，ZIPA test PER26.45→24.85，POWSM<unk>42.20→41.03。该归并不足以解释POWSM旧结果与英语条件之间约15个百分点的差距。原生IPA敏感性评分同样没有把POWSM<unk>变成低错误率；它不是纯粹ARPAbet字符统计错误。','',
'为保持主表评分契约，不只给ZIPA/POWSM单独改用有利归并。rawIPA和额外元音归并仅作诊断，详见audit/SCORE_SENSITIVITY.json，后者不得作为benchmark。','',
'## 4. 主表与比较范围','',
'当前主表的POWSM识别/无文本列更新为<eng>。先前POWSM FA-Bench结果在<unk>下生成，输入为文字G2P，与此行不同条件；已从<eng>行移除FA三列，保留在历史FA报告，不将其拼接成同一实验。','',
'ZIPA和POWSM本次均未做TIMIT微调，OTTC/CIF对应模型则在TIMIT监督训练。这些表不是等训练条件消融，也未审计公开模型的训练集重叠。剩余约26–27%不能仅凭预期判定为实现错误，但应结合参考映射、phone inventory与训练条件解释。','',
'## 5. 完成证据','',
'诊断作业3674292.opbs（18句dev×4条件）和完整英语推理3674299.opbs均F/Exit_status=0。英语dev400/test192预测数量和ID唯一性、预测/逐句评分/指标hash一致性均核验通过；CPU评分成功，权重hash未变。推理保留powsm_eng/下的完整原生IPA与逐帧token路径，评分保留共同ARPAbet假设。','']
(HERE/'audit/REPORT_ZH.md').write_text('\n'.join(L)+'\n')
(HERE/'REVIEW_STATUS.json').write_text(json.dumps({'status':'reviewed','POWSM_condition':'English <eng> reported as known-language condition; <unk> retained as valid control','test_PER_eng':eng['test']['PER'],'test_PFER_eng':eng['test']['PFER'],'ZIPA_status':'historical PER corroborated; no decoding fault found in current audit','audit_report':str(HERE/'audit/REPORT_ZH.md')},ensure_ascii=False,indent=2)+'\n')
# Replace only the condition-specific POWSM row and remove incompatible old FA metrics.
for split in ['dev','test']:
 p=O/f'TIMIT_{split.upper()}_TABLE.csv'
 with p.open(encoding='utf-8-sig') as f:reader=csv.DictReader(f);cols=reader.fieldnames;rows=list(reader)
 x=eng[split];b=x['boundaries']['20']
 for r in rows:
  if r['方法'].startswith('POWSM'):
   r.update({'方法':'POWSM-CTC（<eng>）§','PER':f"{x['PER']:.2f}",'PFER':f"{x['PFER']:.2f}",'无文本 F1':f"{b['F1']:.2f}",'无文本 R-value':f"{b['R_value']:.2f}",'FA F1':'—','FA R-value':'—','MAE (ms)':'—'})
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 p=O/'TIMIT_TABLES_ZH.md';s=p.read_text();marker='## Test（192句）' if split=='test' else '## Dev（400句）';start=s.index('| 方法',s.index(marker));end=s.index('\n\n',start);table='\n'.join(['| '+' | '.join(cols)+' |','|---|'+'|'.join(['---:']*7)+'|']+['| '+' | '.join(r[c] for c in cols)+' |' for r in rows]);s=s[:start]+table+s[end:];p.write_text(s)
p=O/'TIMIT_TABLES_ZH.md';s=p.read_text().replace('**核验状态更新：ZIPA/POWSM 新识别列待独立解码与原生 IPA 评分核验；暂不用于模型排名。**','**核验更新：POWSM识别使用已知英语<eng>条件；原<unk>结果保留为对照。ZIPA结果与历史一致。详见下方复核报告。**')
s=s.replace('§ POWSM-CTC 的PER/PFER/无文本列来自新audio-only评测；FA列为历史clean、参考文字→Charsiu G2P→CTC FA，F1用FA-Bench context/time、MAE用manner matching；未提供FA R-value，dev FA未评测。旧G2P PER未填入识别列。','§ POWSM-CTC 使用已知英语<eng>/<pr>、audio-only解码；<unk>对照test PER/PFER为42.20/14.08，<eng>为27.23/11.68。历史FA使用<unk>及文字输入，未混入本行；<eng> FA尚未评测。')
s+=f'\n## 高PER复核结果\n\n[完整复核报告]({HERE}/audit/REPORT_ZH.md)。POWSM <eng> dev/test均已完整完成；主表已更新，旧<unk>值不删除。\n';p.write_text(s)
p=O/'RESULTS.json';d=load(p)
for r in d['records']:
 if r.get('section')=='TIMIT纯音频识别补测' and r['model']=='POWSM-CTC':r['language_condition']='<unk>';r['reporting_role']='control'
for split,n in [('dev',400),('test',192)]:
 x=eng[split];b=x['boundaries']['20'];d['records'].append(dict(section='TIMIT英语条件识别复核',model='POWSM-CTC',language_condition='<eng>',reporting_role='known_language_primary',dataset='TIMIT',split=split,n=n,mode='text_free',PER=x['PER'],PFER=x['PFER'],F1=b['F1'],R_value=b['R_value'],status='completed',source=str(HERE/'powsm_eng'/f'{split}.json')))
d['powsm_high_PER_review']=str(HERE/'audit/REPORT_ZH.md');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');cols=list(dict.fromkeys(k for r in d['records'] for k in r))
with (O/'RESULTS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(d['records'])
for p in [O/'REPORT_ZH.md',HERE/'REPORT_ZH.md']:
 s=p.read_text();s+='\n## 更正：POWSM语言条件已复核\n\n此前POWSM42.20/14.08来自默认<unk>条件。已知英语<eng>条件的完整test PER/PFER为27.23/11.68，dev为24.13/10.47。主表现用<eng>，历史<unk>和其FA结果保留作条件对照，不混用FA列。完整原因与证据见[复核报告]('+str(HERE/'audit/REPORT_ZH.md')+')。\n';p.write_text(s)
print('Audit closed; English-conditioned results and main tables updated.')
