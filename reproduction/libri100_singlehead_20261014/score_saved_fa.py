"""Score complete saved K2 split paths while the remaining inference continues."""
import bootstrap
from bootstrap import HERE
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    # Use this exact evaluator, avoiding the unrelated installed evaluate package.
    spec = importlib.util.spec_from_file_location('libri100_saved_fa_evaluator', HERE/'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    source_hash = hashlib.sha256((HERE/'evaluate.py').read_bytes()).hexdigest()
    objectives = [('ce', 'CE'), ('jsd_upstream', '原上游 JSD'), ('jsd_standard', 'standard JSD')]
    result = dict(protocol='Whole completed splits only; fixed epoch40; saved native K2 supplied-phone FA; no new inference',
                  evaluator_sha256=source_hash, splits={}, pending=[])
    lines = ['# Libri100 单头：已完成整套 split 的 K2-FSA FA', '',
             '阶段性报告：只评分已覆盖全部参考句子的 split，不把正在写入的部分集合当作完整测试。三组最终总报告另见 REPORT_ZH.md。', '',
             '给定真实音素序列，无参考时间戳；20ms 模型。全集一对一边界 F1/R，MAE 使用同一 split 当前已完成模型共同成功对齐的句子。Libri参考为已有自动对齐，Buckeye为人工标注零样本测试。', '',
             '| Split | OTTC代价 | K2 FA F1@20ms ↑ | R-value ↑ | 共同子集边界 MAE ↓ | 对齐失败 |',
             '|---|---|---:|---:|---:|---:|']
    for split in evaluator.EVALUATION_SPLITS:
        manifest = HERE/'data'/(split+'.jsonl')
        references = {r['utt']: r for r in map(json.loads, manifest.read_text().splitlines())}
        available = {}; metrics = {}
        for objective, label in objectives:
            directory = evaluator.model_directory(objective)
            output = HERE/'evaluation'/objective
            path = output/(split+'_predictions.jsonl')
            if not path.exists() or not (directory/'TRAIN_COMPLETE.json').exists():
                result['pending'].append([split, objective]); continue
            raw = path.read_text().splitlines()
            if len(raw) != len(references):
                result['pending'].append([split, objective]); continue
            try:
                rows = [json.loads(line) for line in raw]
            except json.JSONDecodeError:
                result['pending'].append([split, objective]); continue
            records = {r['utt']: r for r in rows}
            assert len(records) == len(rows) and set(records) == set(references)
            assert hashlib.sha256((output/'evaluator_snapshot.py').read_bytes()).hexdigest() == source_hash
            counts = evaluator.new_counts(); absolute_errors = []
            for utt, row in records.items():
                reference = references[utt]
                assert row['reference'] == reference['reference'] and row['duration'] == reference['duration']
                assert row['wav'] == reference['wav'] and not row['reference_times_used_by_aligner']
                fa = row['forced_k2']; counts['utterances'] += 1
                if 'failure' in fa:
                    counts['alignment_failures'] += 1
                    evaluator.score(counts, row['reference'], [], row['duration'])
                    continue
                collapsed = []; previous = None
                for token in fa['frame_token_ids']:
                    if token != previous and token != 0:
                        collapsed.append(token)
                    previous = token
                assert collapsed == fa['supplied_target_ids']
                spans = fa['predicted_spans']
                assert [p for p, _, _ in spans] == [p for p, _, _ in row['reference']]
                evaluator.score(counts, row['reference'], [t for _, s, e in spans for t in (s, e)], row['duration'])
                errors = counts['errors']
                for (_, rs, re), (_, ps, pe) in zip(row['reference'], spans):
                    errors['tokens'] += 1
                    errors['start'] += abs(rs-ps); errors['end'] += abs(re-pe)
                    absolute_errors.extend([1000*abs(rs-ps), 1000*abs(re-pe)])
                    errors['start_hit20'] += abs(rs-ps) <= .02+1e-9
                    errors['end_hit20'] += abs(re-pe) <= .02+1e-9
            available[objective] = records
            metrics[objective] = evaluator.summary(counts)
            if absolute_errors:
                absolute_errors.sort(); n = len(absolute_errors)
                metrics[objective]['boundary_error_distribution_ms'] = dict(
                    samples=n, mean=statistics.mean(absolute_errors),
                    median=statistics.median(absolute_errors),
                    p90=absolute_errors[int(.9*(n-1))], p99=absolute_errors[int(.99*(n-1))],
                    over_200ms_percent=100*sum(e>200 for e in absolute_errors)/n,
                    over_1s_percent=100*sum(e>1000 for e in absolute_errors)/n)
        if not available:
            continue
        common = [utt for utt in references if all('failure' not in records[utt]['forced_k2'] for records in available.values())]
        for objective, records in available.items():
            error = 0.; tokens = 0
            for utt in common:
                row = records[utt]
                for (_, rs, re), (_, ps, pe) in zip(row['reference'], row['forced_k2']['predicted_spans']):
                    error += abs(rs-ps)+abs(re-pe); tokens += 1
            metrics[objective]['common_subset_boundary_MAE_ms'] = 500*error/tokens if tokens else None
        result['splits'][split] = dict(expected_utterances=len(references),
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            available_objectives=list(available), common_successful_utterances=len(common), metrics=metrics)
        for objective, label in objectives:
            if objective not in metrics:
                continue
            m = metrics[objective]; b = m['boundaries']['20']; mae = m['common_subset_boundary_MAE_ms']
            mae_text = f'{mae:.2f} ms' if mae is not None else '无共同子集'
            lines.append(f"| {split} | {label} | {b['f1_percent']:.2f}% | {b['r_value_percent']:.2f}% | {mae_text} | {m['alignment_failures']}/{m['utterances']} |")
    lines += ['', '每个 split 当前完成的模型及共同子集规模、P/R、10/20/30ms 指标和路径审计结果保存在 FA_COMPLETED_SPLITS.json。所有 split 推理完成后，原评测流程统一上传完整指标至 W&B。', '']
    for filename, text in [('FA_COMPLETED_SPLITS.json', json.dumps(result, indent=2, ensure_ascii=False)+'\n'),
                           ('FA_COMPLETED_SPLITS_ZH.md', '\n'.join(lines))]:
        path = HERE/filename; temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(text); temporary.replace(path)
    print('\n'.join(lines[6:]), flush=True)


if __name__ == '__main__':
    main()
