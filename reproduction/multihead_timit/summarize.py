"""Summarize only genuinely completed forty-epoch four-head experiments."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
objectives=['ctc','ce','jsd_upstream']
head_names={'ottc_arpabet':'OTTC ARPAbet','ifmdd_arpabet':'IF-MDD ARPAbet','mfa_ipa':'MFA IPA','zipa_ipa':'ZIPA IPA127'}
results={}
rows=[]
for objective in objectives:
    directory=ROOT/'models'/objective
    completion_path=directory/'TRAIN_COMPLETE.json'
    if not completion_path.exists():continue
    complete=json.loads(completion_path.read_text())
    if complete['smoke'] or complete['epochs']<40:continue
    metrics=json.loads((directory/'test_metrics.json').read_text())
    identity=json.loads((directory/'wandb_run.json').read_text())
    results[objective]={'completion':complete,'test':metrics,'wandb':identity}
    for head,name in head_names.items():
        kind='token_error_percent' if head=='zipa_ipa' else 'PER_percent'
        error=metrics[f'test_{kind}/{head}']
        f1=metrics.get(f'test_boundary_F1_20ms/{head}')
        boundary=f'{f1:.2f}' if f1 is not None else '无参考边界'
        rows.append(f'| {objective} | {name} | {error:.2f} | {boundary} |')
(ROOT/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
lines=['# TIMIT 四头结果','',f'完成正式训练的配置：{len(results)}/3。','',
       '| Objective | Head | 原生识别错误率 % | 边界 F1@20ms % |',
       '|---|---|---:|---:|',*rows,'',
       'ZIPA 的识别错误率按 token 计算。MFA 使用词典发音目标；两 IPA 头没有真实时间参考。',
       '各头原生识别错误率的标签粒度不同，不能直接比较。测试集为 TIMIT core 192。','']
for objective,result in results.items():lines.append(f'- {objective}: {result["wandb"]["url"]}')
(ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
print(f'Summarized {len(results)}/3 completed experiments')
