"""Completed-only three-seed summaries for each candidate and evaluation split."""
import csv
import fcntl
import json
import statistics
from reproduction.buckeye_frame_sweep_20261027 import common
from reproduction.timit_frame_ce_20261026.report import metrics


def main():
    with (common.HERE/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=[]
        pending=[]
        groups={}
        counts={'dev':6417,'test':6839,'timit_dev':400,'timit_test':192}
        for name in common.matrix():
            output=common.HERE/'models'/name
            if not (output/'BENCHMARK_COMPLETE.json').exists():
                pending.append(name)
                continue
            summary=json.loads((output/'FINAL_SUMMARY.json').read_text())
            for candidate in ('best_dev','final_5','final_10','dev_selected'):
                actual=summary['selected'] if candidate=='dev_selected' else candidate
                for split,result in summary['candidates'][actual].items():
                    assert result['utterances']==counts[split]
                    row=dict(cost=summary['cost'],seed=summary['seed'],candidate=candidate,split=split,**metrics(result))
                    entries.append(row)
                    groups.setdefault((candidate,split,summary['cost']),[]).append(row)
        summaries=[]
        lines=['# Buckeye full40 Frame-wise CE / 标准 JSD / AG / J-div','',
            f'全部评估完成：{12-len(pending)}/12 组；四任务，每任务三节点/三卡，seed42/43/44 独立训练。','',
            '使用 gold 时间标注训练，监督强于 OT-only；不同训练监督的结果分开解释。',
            '主比较为 FP32 single best-dev；final5/final10 和 dev_selected 单列。只按 Buckeye dev PER 选模，test 和 TIMIT 不参与选择。',
            '参数、每轮权重及平均均为 FP32；encoder BF16；最终推理 FP32。评估、预测和报告仅本地保存。',
            'FA：给定映射后音素序列、无剪枝精确标准 CTC Viterbi、blank previous、duration/frame_count；所有样本保留。',
            '边界：20ms 容差、唯一端点、排除句首尾20ms、一对一匹配、冻结 IF-MDD R-value。Text-free 与 FA 分开。',
            '表中为三种子均值 ± 样本标准差，未齐三 seed 的行不计算平均。TIMIT 为跨域评估。','']
        for candidate in ('best_dev','final_5','final_10','dev_selected'):
            for split in counts:
                lines += [f'## {candidate} / {split}','',
                    '| 损失 | 完成 seeds | PER % | PFER % | Text-free F1 % | Text-free R-value % | FA F1 % | FA R-value % | FA MAE ms |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
                for cost in common.COSTS:
                    rows=groups.get((candidate,split,cost),[])
                    complete=sorted(r['seed'] for r in rows)==list(common.SEEDS)
                    record=dict(candidate=candidate,split=split,cost=cost,n=len(rows),complete=complete)
                    cells=[cost,f'{len(rows)}/3']
                    if complete:
                        record['metrics']={key:dict(mean=statistics.mean(r[key] for r in rows),
                            sample_std=statistics.stdev(r[key] for r in rows)) for key in metrics(result)}
                    for key in ['PER','PFER','text_free_F1','text_free_R','FA_F1','FA_R','FA_MAE_ms']:
                        values=record.get('metrics',{}).get(key)
                        cells.append(f"{values['mean']:.2f} ± {values['sample_std']:.2f}" if values else '待齐三种子')
                    lines.append('| '+' | '.join(cells)+' |')
                    summaries.append(record)
                lines += ['']
        lines += ['完整 P/R 与每 seed 明细见 [RESULTS.csv](RESULTS.csv)；[提交记录](SUBMISSIONS_GROUPED.json)；[状态](STATUS.json)。','']
        common.write(common.HERE/'RESULTS.json',dict(complete=not pending,pending=pending,
            summaries=summaries,results=entries,evaluation_uploads=False))
        csv_path=common.HERE/'RESULTS.csv'
        temporary=csv_path.with_suffix('.csv.tmp')
        with temporary.open('w',newline='') as stream:
            fields=list(entries[0]) if entries else ['cost','seed','candidate','split','PER','PFER','FA_MAE_ms']
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
        temporary.replace(csv_path)
        target=common.HERE/'REPORT_ZH.md'
        staging=target.with_suffix('.md.tmp')
        staging.write_text('\n'.join(lines))
        staging.replace(target)
        print(f'REPORT:completed {12-len(pending)}/12',flush=True)


if __name__=='__main__':
    main()
