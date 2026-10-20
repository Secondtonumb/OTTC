"""Summarize bounded GPU cost probes from saved results and continuous telemetry."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SUITE = Path('/work/gm64/m64000/OTTC/reproduction/english_suite')


def samples(path, windows):
    values = []
    if not path.exists():
        return values
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                stamp = datetime.strptime(row['timestamp'].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                if not any(first <= stamp <= last for first, last in windows):
                    continue
                values.append({'util': float(row[' utilization.gpu [%]'].strip().split()[0]),
                               'power': float(row[' power.draw [W]'].strip().split()[0])})
            except (KeyError, ValueError):
                continue
    return values


def telemetry(directory, records, gpus, warmup):
    windows = [(r['start_unix'], r['end_unix']) for r in records if r['step'] >= warmup]
    output = {}
    for rank in range(gpus):
        path = directory/f'gpu_rank{rank}.csv' if gpus > 1 or (directory/'gpu_rank0.csv').exists() else directory/'gpu.csv'
        selected = samples(path, windows)
        output[str(rank)] = {'samples': len(selected),
                             'mean_gpu_util_percent': statistics.mean(v['util'] for v in selected) if selected else None,
                             'median_gpu_util_percent': statistics.median(v['util'] for v in selected) if selected else None,
                             'mean_power_W': statistics.mean(v['power'] for v in selected) if selected else None}
    return output


def main():
    registry_path = SUITE/'GPU_COST_JOBS.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    mfa_job = registry.get('latest_mfa_job', '3512870.opbs')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mfa-dir', type=Path, default=SUITE/f'gpu_cost_probe_{mfa_job}')
    parser.add_argument('--libri-dir', type=Path, default=SUITE/'gpu_cost_libri_3512871.opbs')
    parser.add_argument('--output', type=Path, default=SUITE/'GPU_COST_TEST_RESULTS.md')
    args = parser.parse_args()
    payload = {'updated_at': datetime.now().astimezone().isoformat(), 'mfa': [], 'libri': [], 'status': {},
               'job_registry': registry}
    lines = ['# GPU 成本短测', '', '更新时间：' + payload['updated_at'], '',
             '测试上限合计 3 GPU 小时，不自动续投。当前报告只读取已生成的结果。', '',
             'MFA 固定每一步的全局样本 ID 和音频量，改变实际计算的 GPU 数量；',
             '采用当前优化 microbatch，直接加载隔离副本的模型/优化器，不验证正式 checkpoint 的 sampler/RNG 迁移。',
             'MFA 测速分配共八卡；小卡数阶段的空闲卡属于本次测试成本，候选成本按实际计算卡数归一化，',
             '不能把候选 GPU 小时之和当成本次 PBS 的实际计费。', '',
             '## MFA', '']
    mfa_path = args.mfa_dir/'results.json'
    complete = args.mfa_dir/'BENCHMARK_COMPLETE.json'
    payload['status']['mfa_complete'] = json.loads(complete.read_text()) if complete.exists() else None
    if mfa_path.exists():
        groups = defaultdict(list)
        for result in json.loads(mfa_path.read_text()):
            result['telemetry'] = telemetry(args.mfa_dir, result.get('records', []), result['gpus'], 2)
            payload['mfa'].append(result)
            groups[(result['arm'], result['gpus'])].append(result)
        lines += ['| 模型 | GPU | 有效重复 | 秒/步 | GPU 秒/音频小时 | 相对八卡成本 | 连续采样平均利用率 | 峰值 GiB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for (arm, count), candidates in sorted(groups.items()):
            valid = [r for r in candidates if r['status'] == 'PASS']
            baseline = [r['gpu_seconds_per_audio_hour'] for r in groups.get((arm, 8), []) if r['status'] == 'PASS']
            if not valid:
                lines.append(f'| {arm} | {count} | 0 | FAIL | FAIL | - | - | - |')
                continue
            cost = statistics.mean(r['gpu_seconds_per_audio_hour'] for r in valid)
            ratio = f'{cost/statistics.mean(baseline):.3f}' if baseline else '-'
            util = [t['mean_gpu_util_percent'] for r in valid for t in r['telemetry'].values()
                    if t['mean_gpu_util_percent'] is not None]
            util_label = f'{statistics.mean(util):.1f}%' if util else '-'
            lines.append(f'| {arm} | {count} | {len(valid)} | {statistics.mean(r["mean_step_seconds"] for r in valid):.3f} | '
                         f'{cost:.3f} | {ratio} | {util_label} | {max(r["peak_GiB"] for r in valid):.2f} |')
        lines += ['', '每档两次独立重置，正反测试顺序；每次十步、前两步预热。只代表这些固定 batch。',
                  '各档 microbatch 切分和随机 mask 可能不同，不是逐位等价或收敛质量测试。', '']
    else:
        lines += ['尚无测量结果；以 PBS 状态为准。', '']
    for record in registry.get('jobs', {}).values():
        if record.get('exit_status') and record.get('failure'):
            lines += [f'历史失败作业 {record["job"]}：{record["failure"]}；'
                      f'运行 {record["walltime"]}，实际 {record["gpu_hours"]:.4f} GPU 小时，没有卡数比较结果。', '']
    lines += ['## LibriSpeech', '']
    libri_path = args.libri_dir/'results.json'
    complete = args.libri_dir/'BENCHMARK_COMPLETE.json'
    payload['status']['libri_complete'] = json.loads(complete.read_text()) if complete.exists() else None
    if libri_path.exists():
        results = json.loads(libri_path.read_text())
        lines += ['| 模型 | 输入 | microbatch | 重计算 | 秒/步 | 相对 micro=8 重计算成本 | 连续采样平均利用率 | 峰值 GiB |',
                  '|---|---|---:|---|---:|---:|---:|---:|']
        baseline = {(r['objective'], r['fixture']): r['mean_step_seconds'] for r in results
                    if r['status'] == 'PASS' and r['microbatch'] == 8 and r['gradient_checkpointing']}
        for result in results:
            result['telemetry'] = telemetry(args.libri_dir, result.get('records', []), 1, 1)
            payload['libri'].append(result)
            util = result['telemetry']['0']['mean_gpu_util_percent']
            util_label = f'{util:.1f}%' if util is not None else '-'
            common = f'| {result["objective"]} | {result["fixture"]} | {result["microbatch"]} | {"开" if result["gradient_checkpointing"] else "关"} | '
            if result['status'] != 'PASS':
                lines.append(common + 'FAIL | - | - | - |')
            else:
                ratio = result['mean_step_seconds']/baseline[(result['objective'], result['fixture'])]
                lines.append(common + f'{result["mean_step_seconds"]:.3f} | {ratio:.3f} | {util_label} | {result["peak_GiB"]:.2f} |')
        lines += ['', '每档从同一模型状态开始，共七次更新、首步预热；有效 batch=64，真实 Libri960 典型/最长输入。',
                  '包含 collate、复制、四头 loss、backward、梯度裁剪与 AdamW，音频预读取；不包含整轮加载/评估。',
                  'CE/JSD 使用已冻结对齐分支的现有模型，仅适用于该阶段；改变切分可能改变 masked head 的有效权重。', '']
    else:
        lines += ['尚无测量结果；以 PBS 状态为准。', '']
    lines += ['## 证据', '', f'- MFA：`{args.mfa_dir}`', f'- LibriSpeech：`{args.libri_dir}`',
              '- 各目录保留结果、逐卡 CSV、checkpoint 副本、来源 hash、源码快照及 W&B run。',
              '- 最终通过/失败以 BENCHMARK_COMPLETE.json 与 PBS exit status 共同判断，部分结果不代表测试已完成。',
              '- 最长真实输入通过短测不代表整个语料永不 OOM；正式配置暂未改动。', '']
    args.output.write_text('\n'.join(lines))
    args.output.with_suffix('.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
