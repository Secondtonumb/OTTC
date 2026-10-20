"""Refresh a local, timestamped PBS/checkpoint status report for the English suite."""
import argparse
import ast
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent
PBS_SNAPSHOT = None


def job_status(job):
    if PBS_SNAPSHOT is not None:
        return PBS_SNAPSHOT['jobs'].get(job.split('.')[0], {'job': job, 'query_ok': False})
    process = subprocess.run(['qstat', '-f', job], capture_output=True, text=True, timeout=30)
    if process.returncode or not re.search(r'^\s*job_state = ',process.stdout,re.M):
        process = subprocess.run(['qstat', '-f', '-H', job], capture_output=True, text=True, timeout=30)
    fields = {}
    output = process.stdout.replace('\n\t', '')
    for key in ['job_state', 'queue', 'estimated.start_time', 'stime', 'obittime', 'Exit_status', 'comment',
                'Resource_List.ngpus', 'resources_used.walltime', 'resources_used.gpu_avg_util_per_nodes',
                'resources_used.gpu_max_memory_per_nodes']:
        match = re.search(r'^\s*'+re.escape(key)+r' = (.*)$', output, re.M)
        if match:
            fields[key] = match.group(1)
    return {'job': job, **fields, 'query_ok': process.returncode == 0}


def tail_text(path, budget=220000):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-budget))
        return stream.read().decode(errors='replace')


def live_progress(directory, mfa=False):
    log = directory/('updates.jsonl' if mfa else 'train.log')
    text = tail_text(log)
    if not text:
        return {}
    entries = []
    for line in re.findall(r'\{[^\r\n]+\}', text):
        try:
            value = json.loads(line) if mfa else ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            entries.append(value)
    result = {'log_modified_at': datetime.fromtimestamp(log.stat().st_mtime).astimezone().isoformat(timespec='seconds')}
    if mfa:
        result['last_update'] = next((e for e in reversed(entries) if e.get('event') == 'update'), {})
        config = directory/'run_config.json'
        if config.exists():
            result['estimated_epoch_steps'] = json.loads(config.read_text())['contract']['estimated_epoch_steps']
    else:
        steps = re.findall(r'(\d+)/(\d+) \[', text)
        if steps:
            result.update(step=int(steps[-1][0]), total_steps=int(steps[-1][1]))
        result['last_train'] = next((e for e in reversed(entries) if 'loss' in e), {})
        result['last_dev'] = next((e for e in reversed(entries) if 'eval_loss' in e), {})
    return result


def mfa_development(arm, directory=None):
    latest = (directory or SUITE/'mfa_models'/arm)/'latest_dev_metrics.json'
    if latest.exists():
        record = json.loads(latest.read_text())
        return {'source': str(latest), 'stage': '正式逐轮 dev',
                'optimizer_step': record['optimizer_step'], 'text_free': record['text_free']}
    paths = sorted((SUITE/'mfa_evaluation'/arm).glob('step-*/metrics.json'),
                   key=lambda path: int(path.parent.name[5:]), reverse=True)
    for path in paths:
        if not (path.parent/'EVAL_COMPLETE.json').exists():
            continue
        record = json.loads(path.read_text())
        dev = record.get('splits', {}).get('development', {})
        if dev and not dev.get('limit', 0):
            return {'source': str(path), 'stage': '最终 checkpoint dev' if record['final'] else '固定中途 checkpoint 诊断',
                    'optimizer_step': record['progress']['global_step'], 'text_free': dev['text_free']}
    return None


def main():
    submission = json.loads((SUITE/'submission_status.json').read_text())
    extra_runs_path = SUITE/'additional_tracking_runs.json'
    extra_runs = json.loads(extra_runs_path.read_text()) if extra_runs_path.exists() else {}
    jobs = {'LibriSpeech 第一段': submission['libri']['train_segments'][0],
            'LibriSpeech 第二段': submission['libri']['train_segments'][1],
            'LibriSpeech K2 / text-free 评估': submission['libri']['evaluation_jobs'][0],
            **{'MFA '+arm: job for arm, job in submission['mfa_english']['production_jobs'].items()}}
    for name, record in extra_runs.items():
        jobs[name] = record['job']
        if record.get('continuation_job'):
            jobs[name+' continuation'] = record['continuation_job']
        for key in ['latest_training_job','evaluation_job']:
            if record.get(key):jobs[name+' '+key]=record[key]
    for path in [SUITE/'continuation_status.json', *SUITE.glob('mfa_models/*/segment_status.json')]:
        if path.exists():
            event = json.loads(path.read_text())
            for key in ['continuation_job', 'evaluation_job']:
                if event.get(key):
                    jobs[str(path.relative_to(SUITE))+' '+key] = event[key]
    for key in ['libri_probe_job','libri_frozen_probe_job']:
        if submission.get('throughput',{}).get(key):
            jobs['吞吐验证 '+key]=submission['throughput'][key]
    for key in ['mfa_prepare_job','mfa_current_dev_test_job','mfa_inference_gate_job']:
        if submission.get('evaluation_updates',{}).get(key):
            jobs['评测补齐 '+key]=submission['evaluation_updates'][key]
    for arm,job in submission.get('evaluation_updates',{}).get('mfa_current_eval_jobs',{}).items():
        jobs['MFA '+arm+' current dev/test']=job
    comparisons=ROOT/'reproduction/checkpoint_comparison/submissions.json'
    if comparisons.exists():
        for name,job in json.loads(comparisons.read_text()).items():
            jobs['三个 checkpoint 比较 '+name]=job
    extra_path=SUITE/'additional_tracking_jobs.json'
    if extra_path.exists():
        jobs.update(json.loads(extra_path.read_text()))
    cost_jobs = SUITE/'GPU_COST_JOBS.json'
    if cost_jobs.exists():
        for name, record in json.loads(cost_jobs.read_text())['jobs'].items():
            jobs['GPU 成本短测 '+name] = record['job']
    statuses = {name: job_status(job) for name, job in jobs.items()}
    states = {'Q': '排队', 'H': '等待前置作业', 'R': '运行中', 'F': '已退出', 'E': '退出中'}
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    gate_path = SUITE/'MFA_GATE_PASS.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    runs = {}
    for dataset in ['libri100', 'libri960']:
        for objective in ['ctc', 'ce', 'jsd_upstream']:
            directory = ROOT/'reproduction'/f'multihead_{dataset}'/'models'/objective
            runs[dataset+'/'+objective] = {name: json.loads((directory/name).read_text()) for name in
                ['TRAIN_COMPLETE.json', 'CHECKPOINT_PAUSED.json', 'wandb_run.json'] if (directory/name).exists()}
            runs[dataset+'/'+objective]['live'] = live_progress(directory)
    mfa_directories = {arm: SUITE/'mfa_models'/arm for arm in submission['mfa_english']['production_jobs']}
    mfa_directories.update({name: Path(record['directory']) for name, record in extra_runs.items() if record.get('directory')})
    for arm, directory in mfa_directories.items():
        runs['mfa/'+arm] = {name: json.loads((directory/name).read_text()) for name in
            ['TRAIN_COMPLETE.json', 'progress.json', 'segment_status.json', 'wandb_run.json'] if (directory/name).exists()}
        runs['mfa/'+arm]['live'] = live_progress(directory, mfa=True)
        runs['mfa/'+arm]['development'] = mfa_development(arm, directory)
    for name, result in runs.items():
        if name.startswith('libri'):
            job = submission['libri']['train_segments'][1]
        else:
            arm = name.split('/', 1)[1]
            extra = extra_runs.get(arm, {})
            job = result.get('segment_status.json', {}).get('continuation_job') or extra.get('continuation_job') or extra.get('latest_training_job') or extra.get('job') or submission['mfa_english']['production_jobs'][arm]
        result['current_job'] = job
        state = job_status(job).get('job_state')
        result['pbs_state'] = state
        started = bool(result.get('progress.json') or result.get('CHECKPOINT_PAUSED.json') or result['live'].get('last_train'))
        result['training_state'] = ('已完成' if 'TRAIN_COMPLETE.json' in result else
                                    '运行中' if state == 'R' else
                                    '已保存，续训排队' if started and state in ['Q', 'H'] else
                                    '尚未启动，排队' if state in ['Q', 'H'] else
                                    (extra.get('training_state_override') or '已退出，未完成') if state == 'F' else '需核对')
    jsd_smoke_path=SUITE/'mfa_jsd/validation/SMOKE_PASS.json'
    jsd_smoke=json.loads(jsd_smoke_path.read_text()) if jsd_smoke_path.exists() else {}
    snapshot = {'captured_at': now, 'pbs': statuses, 'mfa_gpu_gate': gate, 'production_runs': runs,
                'zapformer_jsd': {'gpu_validation':jsd_smoke,'formal_run_started':(SUITE/'mfa_models/ottc_jsd_upstream/run_config.json').exists()}}
    if PBS_SNAPSHOT:
        snapshot['pbs_captured_at'] = PBS_SNAPSHOT['captured_at']
    (SUITE/'tracking.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 英语实验进度', '', '检查时间：'+now+'。这是实际查询的快照。', '',
             'LibriSpeech 全量数据审计与三组官方 LV60 GPU smoke 已通过。',
             'MFA 四组真实更新与 CTC / FiLM 保存后恢复验证：'+('PASS。' if gate else '等待完成。'),
             'Libri100 三组已完成 40 轮；Libri960 是否运行以续训 Job 和保存标记为准，首段 exit0 不代表 40 轮全部完成。', '',
             '| 作业 | Job ID | 当前状态 | 实际 / 预计启动（JST） |', '|---|---|---|---|']
    for name, status in statuses.items():
        state = states.get(status.get('job_state'), '未知')
        if status.get('job_state') == 'F':
            state += '（exit='+status.get('Exit_status', '未知')+'）'
        started=status.get('stime')
        start_text='实际：'+started if started else '预计：'+status['estimated.start_time'] if status.get('estimated.start_time') else '—'
        lines.append('| '+' | '.join([name, status['job'], state,start_text])+' |')
    lines.extend(['', '预计启动时间会随调度变化。等待前置作业不是训练停滞；正式 run 启动后才有正式 loss 曲线。', '',
                  '## 正式训练进度', '',
                  '| 模型 | 最近实际步数 | 轮次 | 最近记录 train loss | W&B |', '|---|---:|---|---:|---|'])
    for name, result in runs.items():
        identity = result.get('wandb_run.json', {})
        progress = result.get('TRAIN_COMPLETE.json', result.get('CHECKPOINT_PAUSED.json', result.get('progress.json', {})))
        live = result['live']
        train = live.get('last_train', live.get('last_update', {}))
        step = live.get('step', train.get('step', progress.get('global_step', progress.get('steps', '—'))))
        epoch = train.get('epoch')
        if 'TRAIN_COMPLETE.json' in result:
            completed=result['TRAIN_COMPLETE.json']
            epoch_text=f'✅ 已完成 {completed.get("epochs",10)} 轮'
            step=completed.get('steps',completed.get('progress',{}).get('global_step',step))
        elif epoch is not None:
            if 'CHECKPOINT_PAUSED.json' in result and result['pbs_state'] != 'R':
                epoch_text = f'{result["CHECKPOINT_PAUSED.json"]["epochs"]:.2f} / 40（已保存 checkpoint）'
            else:
                epoch_text = f'第 {int(epoch)+1} / 10 轮（epoch 从 0 计）' if name.startswith('mfa/') else f'{epoch:.2f} / 40（最近 loss 记录）'
        else:
            epoch_text = '尚未启动' if not progress else str(progress.get('epochs', progress.get('epoch', '—')))
        if result['training_state'] != '已完成':
            epoch_text = result['training_state']+'；'+epoch_text
        lines.append('| '+' | '.join([name, str(step), epoch_text, f'{train["loss"]:.4f}' if 'loss' in train else '—',
                                      '[W&B]('+identity['url']+')' if identity.get('url') else '—'])+' |')
    lines.extend(['', '步数来自最新日志，loss 按原记录间隔展示；checkpoint 可能早于最新训练步。原始 loss 定义不同，不直接比较量级。', '',
                  '## LibriSpeech 最近 dev（原始 OTTC ARPAbet head，text-free）', '',
                  '| 模型 | Dev epoch | PER % | F1@20ms % |', '|---|---:|---:|---:|'])
    for name, result in runs.items():
        dev = result['live'].get('last_dev', {})
        if dev:
            lines.append(f'| {name} | {dev["epoch"]:.2f} | {dev["eval_PER_percent/ottc_arpabet"]:.2f} | {dev["eval_boundary_F1_20ms/ottc_arpabet"]:.2f} |')
    lines.extend(['', '中途 dev 结果来自不同 epoch，不能当作最终公平比较；LibriSpeech 边界参考是自动对齐。Libri100 checkpoint 比较和独立 JSD FA-bench 已有结果，Libri960 最终 K2 FA 仍等待训练完成。', '',
                  '## MFA 外部 TIMIT dev（text-free）', '',
                  '| 模型 | checkpoint step | 阶段 | PER % | Precision@20ms % | Recall@20ms % | F1@20ms % |',
                  '|---|---:|---|---:|---:|---:|---:|'])
    for arm in mfa_directories:
        record=runs['mfa/'+arm]['development']
        if record:
            metrics=record['text_free']
            lines.append(f'| {arm} | {record["optimizer_step"]} | {record["stage"]} | {metrics["PER_percent"]:.2f} | {metrics["boundary_precision_20ms_percent"]:.2f} | {metrics["boundary_recall_20ms_percent"]:.2f} | {metrics["boundary_F1_20ms_percent"]:.2f} |')
        else:lines.append('| '+arm+' | — | 尚无完整 dev 指标 | — | — | — | — |')
    lines.extend(['', '恢复时及逐轮完整 dev logging 已在当前续训生效。历史缺失的逐轮指标不补造。', '',
                  '[实验完成状态总表](EXPERIMENT_STATUS.md) · [前端区别](FRONTENDS.md) · [全部结果](RESULTS.md) · [三个 checkpoint 的比较](../checkpoint_comparison/README.md)。'])
    lines.extend(['', '## 验证运行 W&B（不代表正式长训）', ''])
    if jsd_smoke:
        lines+=['Zapformer 上游 JSD GPU 验证：'+jsd_smoke.get('status','未知')+'；完整模型 loss/梯度与原 OTTC 归档实现一致。',
                '独立 JSD 长训已启动，当前状态见上表；[实现与验证](mfa_jsd/README.md)。','']
        if jsd_smoke.get('wandb',{}).get('url'):
            lines.append('- Zapformer JSD smoke：[W&B]('+jsd_smoke['wandb']['url']+')')
    for directory in [SUITE/'validation', SUITE/'mfa_validation'/submission['mfa_english']['gate_job']]:
        for identity_path in sorted(directory.glob('*/wandb_run.json')):
            identity = json.loads(identity_path.read_text())
            if identity.get('url'):
                lines.append('- '+str(identity_path.parent.relative_to(SUITE))+': [W&B]('+identity['url']+')')
    lines.extend(['', '## GPU 吞吐调整', '',
                  '[实测协议、吞吐结果与下一段生效配置](THROUGHPUT.md)。', '',
                  '[GPU 成本短测结果与连续利用率](GPU_COST_TEST_RESULTS.md)。', '',
                  '[当前正式续训吞吐与 GPU 采样](GPU_RUNTIME_TRACKING.md)。', '',
                  'Libri100 已用 microbatch=4 完成；Libri960 续训策略为 microbatch=8、effective batch=64。',
                  'MFA 当前 CTC/OTTC/SPOTTC 续训已分别启用限额因子 4/4/2；排队 FiLM 为 4。', '',
                  '刷新：`python reproduction/english_suite/track.py`。', '',
                  '[完整实验配置与调度记录](README.md)。', ''])
    (SUITE/'TRACKING.md').write_text('\n'.join(lines))
    submission['updated_at'] = now
    submission['libri']['train_state'] = runs['libri960/ctc']['training_state']
    submission['mfa_english']['gate_state'] = 'passed' if gate else 'pending'
    submission['latest_tracking'] = snapshot
    (SUITE/'submission_status.json').write_text(json.dumps(submission, ensure_ascii=False, indent=2)+'\n')
    from mark_experiments import main as mark_experiments
    mark_experiments()
    from gpu_runtime_tracking import main as gpu_runtime_tracking
    gpu_runtime_tracking()
    print(json.dumps({'captured_at': now, 'jobs': len(statuses), 'mfa_gate': 'PASS' if gate else 'pending'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbs-snapshot', type=Path, help='使用本轮实际查询的 PBS JSON，避免重复网络查询')
    args = parser.parse_args()
    if args.pbs_snapshot:
        PBS_SNAPSHOT = json.loads(args.pbs_snapshot.read_text())
    main()
