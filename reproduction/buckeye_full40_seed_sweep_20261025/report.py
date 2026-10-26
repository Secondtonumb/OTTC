"""Completed-only three-seed local reports, separate candidates and corpora."""
import fcntl
import json

from reproduction.buckeye_full40_seed_sweep_20261025.common import HERE, COSTS, SEEDS, base
from reproduction.timit_seed_sweep_20261022.report import row_from_metrics, summarize, write_summary


def main():
    with (HERE / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        groups, pending = {}, []
        for cost in COSTS:
            for seed in SEEDS:
                name = f'{cost}__seed{seed}'
                output = HERE / 'models' / name
                if not (output / 'BENCHMARK_COMPLETE.json').exists():
                    pending.append(name)
                    continue
                record = json.loads((output / 'FINAL_SUMMARY.json').read_text())
                for candidate, splits in record['results'].items():
                    for split, result in splits.items():
                        groups.setdefault((candidate, split), []).append(row_from_metrics(cost, seed, output, result))
                for split, result in record['results'][record['selected']].items():
                    groups.setdefault(('dev_selected', split), []).append(row_from_metrics(cost, seed, output, result))
        summaries = {}
        lines = ['# 完整 Buckeye 四损失 × 三种子', '',
                 f'已完成全部评估：{12 - len(pending)}/12 组。未完成组不进入汇总。', '',
                 '原始 SSL 初始化；Buckeye 训练，Buckeye dev 选模型；Buckeye test 和 TIMIT 不参与选择。',
                 'best3 为同一 seed 内 dev PER 最低三轮的 FP32 等权权重平均；不是跨 seed 平均。',
                 '训练40轮，仅保留dev-best三份权重和一份最新续训状态，不做last5/last10；dev_selected 仅按 Buckeye dev FP32 PER 选择。',
                 '边界：20ms容差，唯一端点，排除句首尾20ms，一对一匹配，冻结 IF-MDD R-value；text-free 与 supplied-phone FA 分开。',
                 '表中数值为三种子均值 ± 样本标准差；完整 P/R 和其他指标保存在各 CSV。评估仅本地。', '']
        for (candidate, split), rows in sorted(groups.items()):
            summary = summarize(rows, SEEDS)
            key = f'{candidate}_{split}'
            summaries[key] = summary
            write_summary(HERE / f'{key}_THREE_SEED_SUMMARY.csv', summary)
            lines += [f'## {candidate} / {split}', '',
                      '| 损失 | n | PER % | PFER % | Text-free F1 % | Text-free R % | FA F1 % | FA R % | FA MAE ms |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for record in summary:
                cells = [record['cost'], str(record['n'])]
                for metric in ('PER', 'PFER', 'text_free_F1', 'text_free_R', 'FA_F1', 'FA_R', 'FA_MAE_ms'):
                    values = record['metrics'][metric]
                    cells.append(f"{values['mean']:.2f} ± {values['sample_std']:.2f}" if values['sample_std'] is not None else '待齐三种子')
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines += ['', f'[完整 CSV]({key}_THREE_SEED_SUMMARY.csv)', '']
        base.write(HERE / 'COMPARISON.json', dict(complete=not pending, pending=pending, summaries=summaries,
                                                evaluation_uploads=False))
        temporary = HERE / 'REPORT_ZH.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(HERE / 'REPORT_ZH.md')
        print(f'Completed evaluation: {12 - len(pending)}/12')


if __name__ == '__main__':
    main()