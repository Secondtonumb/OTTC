"""Four independent seeds, sample SD; no means from unfinished training."""
import csv
import fcntl
import json
from pathlib import Path
import statistics

from reproduction.timit_seed_sweep_20261022.report import row_from_metrics,METRICS
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=(42,43,44,45)
TABLE_KEYS=('PER','PFER','text_free_F1','text_free_R','FA_F1','FA_R','FA_MAE_ms')


def main():
    lock=(HERE/'.report.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    rows,pending=[],[]
    for method in ('AG','CROTTC_AG'):
        for seed in SEEDS:
            if method=='CROTTC_AG':
                source=HERE/'models'/f'crottc_ag__seed{seed}'
            elif seed==42:
                source=ROOT/'reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__jsd_upstream'
            else:
                source=ROOT/'reproduction/timit_seed_sweep_20261022/models'/f'ag__seed{seed}'
            path=source/'benchmark_best_dev_test.json'
            if not (source/'BENCHMARK_COMPLETE.json').exists() or not path.exists():
                state=source/'LATEST_STATE.json'
                pending.append(dict(method=method,seed=seed,state=json.loads(state.read_text()) if state.exists() else None))
                continue
            result=json.loads(path.read_text())
            assert result['utterances']==192
            marker=json.loads((source/'TRAIN_COMPLETE.json').read_text())
            assert marker['epoch']>=40-1e-5 and marker['step']==2920
            if 'PFER_percent' not in result['modes']['text_free']:
                # Historical seed42 stays unchanged; this is an in-memory backfill.
                result['modes']['text_free']['PFER_percent']=score_predictions(path.with_suffix('.predictions.jsonl'),result)['PFER_percent']
            rows.append(row_from_metrics(method,seed,source,result))
    summaries=[]
    for method in ('AG','CROTTC_AG'):
        group=[row for row in rows if row['cost']==method]
        summaries.append(dict(method=method,seeds=sorted(row['seed'] for row in group),n=len(group),
            complete=len(group)==4,metrics={key:dict(n=len(group),mean=statistics.mean(row[key] for row in group) if group else None,
                sample_std=statistics.stdev(row[key] for row in group) if len(group)>1 else None) for key in METRICS}))
    report=dict(protocol='same TIMIT data, monophone native10ms, original SSL, last CNN trainable; single best-dev; FP32 inference; n4 sample SD ddof1',
        expected_seeds=list(SEEDS),complete=all(s['complete'] for s in summaries),rows=rows,pending=pending,summaries=summaries)
    temporary=HERE/'COMPARISON.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(HERE/'COMPARISON.json')
    fields=['cost','seed',*METRICS,'source']
    with (HERE/'PER_SEED.csv.tmp').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    (HERE/'PER_SEED.csv.tmp').replace(HERE/'PER_SEED.csv')
    lines=['# TIMIT CROTTC + AG 四 seed 比较','',
        '同协议 single best-dev；四个独立 seed 的均值 ± 样本标准差（ddof=1）。平均权重实验另外保存，不进入本表。','',
        '| 方法 | n | PER | PFER | 无文本 F1 | 无文本 R-value | FA F1 | FA R-value | FA MAE ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for summary in summaries:
        cells=[f"{summary['metrics'][key]['mean']:.2f}±{summary['metrics'][key]['sample_std']:.2f}" for key in TABLE_KEYS] if summary['complete'] else ['待四个 seed 全部完成']+['—']*6
        lines.append('| '+' | '.join([summary['method'],str(summary['n']),*cells])+' |')
    lines += ['',f"尚未完成：{len(pending)} 个模型。",'',
              'CR 配方复用 IF-MDD：共享语速扰动、两路独立 DropFreq/DropChunk、双向 teacher detach KL，权重0.1；保留历史分母及整体按原始句数缩放。','',
              '训练/评估完成分别由 TRAIN_COMPLETE.json 和 BENCHMARK_COMPLETE.json 核实；所有评估预测和指标仅保存在本地。']
    (HERE/'REPORT_ZH.md.tmp').write_text('\n'.join(lines)+'\n')
    (HERE/'REPORT_ZH.md.tmp').replace(HERE/'REPORT_ZH.md')
    print(json.dumps(dict(complete=report['complete'],rows=len(rows),pending=len(pending))),flush=True)


if __name__=='__main__':
    main()
