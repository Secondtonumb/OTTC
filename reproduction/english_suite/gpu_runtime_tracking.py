"""Summarize observed training updates and GPU samples without launching work."""
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from track import SUITE, tail_text


def main():
    snapshot = json.loads((SUITE/'tracking.json').read_text())
    records = {}
    directories = {arm: SUITE/'mfa_models'/arm for arm in ['ctc', 'ottc', 'spottc']}
    extras = json.loads((SUITE/'additional_tracking_runs.json').read_text())
    directories.update({arm: Path(record['directory']) for arm, record in extras.items()})
    for arm, directory in directories.items():
        updates = []
        for line in tail_text(directory/'updates.jsonl', budget=5000000).splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('event') == 'update' and 'step_seconds' in record:
                updates.append(record)
        updates = updates[-500:]
        if not updates:
            continue
        current = snapshot['production_runs']['mfa/'+arm]
        job = current['current_job']
        if current['pbs_state'] in ['Q', 'H'] and arm in extras:
            job = extras[arm]['job']
        result = {'source': str(directory/'updates.jsonl'), 'measured_job': job,
                  'training_state': current['training_state'], 'updates': len(updates),
                  'first_step': updates[0]['step'], 'last_step': updates[-1]['step'],
                  'mean_step_seconds': statistics.mean(u['step_seconds'] for u in updates),
                  'median_step_seconds': statistics.median(u['step_seconds'] for u in updates),
                  'audio_seconds_per_wall_second': sum(u['audio_seconds'] for u in updates)/sum(u['step_seconds'] for u in updates),
                  'peak_allocated_max_GiB': max(u.get('peak_allocated_max', 0) for u in updates)/2**30,
                  'gpu_samples': []}
        for path in sorted(directory.glob('gpu_rank*_'+job+'.csv')):
            with path.open() as stream:
                rows = list(csv.DictReader(stream, skipinitialspace=True))
            samples = []
            for row in rows:
                try:
                    samples.append(float(row['utilization.gpu [%]'].split()[0]))
                except (KeyError, ValueError, IndexError):
                    continue
            if samples:
                result['gpu_samples'].append({'source': str(path), 'count': len(samples),
                                              'first_timestamp': rows[0]['timestamp'],
                                              'last_timestamp': rows[-1]['timestamp'],
                                              'mean_utilization_percent': statistics.mean(samples),
                                              'median_utilization_percent': statistics.median(samples)})
        records[arm] = result
    pbs = {v['job']: v for v in snapshot['pbs'].values()}
    active_gpus = sum(int(v.get('Resource_List.ngpus', 0)) for v in pbs.values() if v.get('job_state') == 'R')
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    result = {'captured_at': now, 'pbs_captured_at': snapshot.get('pbs_captured_at', snapshot['captured_at']),
              'active_training_gpus': active_gpus, 'runs': records,
              'scope': 'Last 500 logged updates; GPU samples include initialization/checkpoint/dev and cover only recorded ranks. No full-epoch speedup claim.'}
    (SUITE/'GPU_RUNTIME_TRACKING.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 正式训练 GPU 与吞吐快照', '', '检查时间：'+now+'。', '',
             f'当前已核对的正式训练共占用 {active_gpus} 张 GPU；排队作业未占用 GPU。', '',
             '| 模型 | 测量 Job | 状态 | 最近步数 | 平均秒/步 | 最大 allocated GiB | GPU 采样平均利用率 |',
             '|---|---|---|---:|---:|---:|---|']
    for arm, record in records.items():
        samples = record['gpu_samples']
        util = (f'{statistics.mean(s["mean_utilization_percent"] for s in samples):.1f}%（{len(samples)} 个 rank）' if samples else '无本地连续采样')
        lines.append(f'| {arm} | {record["measured_job"]} | {record["training_state"]} | {record["last_step"]} | {record["mean_step_seconds"]:.3f} | {record["peak_allocated_max_GiB"]:.2f} | {util} |')
    lines += ['', '秒/步按最后 500 条真实更新计算，不含 dev/checkpoint 暂停；音频批次随训练变化，不能把旧片段和本表当作固定输入对照。',
              'GPU 利用率包含当前片段从采样开始至今的初始化、保存和 dev。CTC/CE/SPOTTC 仅有 rank0 CSV，不能代替八卡整体均值；S3T2 有逐卡采样。', '',
              '3507745 首段正式训练：PBS 记录为三卡 23:05:15，约 69.3 GPU 小时；各卡整段平均利用率 79%、80%、83%。960h 当前续训状态见实验进度表。', '',
              '[固定输入 GPU 数量及 microbatch 对照](GPU_COST_TEST_RESULTS.md) · [吞吐调整依据](THROUGHPUT.md) · [实验进度](TRACKING.md)', '',
              '刷新：`python reproduction/english_suite/gpu_runtime_tracking.py`。', '']
    (SUITE/'GPU_RUNTIME_TRACKING.md').write_text('\n'.join(lines))
    print(json.dumps({'captured_at': now, 'active_training_gpus': active_gpus, 'arms': list(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
