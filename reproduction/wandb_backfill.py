"""Backfill genuine Trainer histories and build an interactive W&B report."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT)]
import wandb
from ottc.utils.wandb_tracking import benchmark_summary, defaults

VARIANTS = [('CTC', 'runexp_ctc_fuse_phoneme'),
            ('OTTC + CE', 'runexp_ottc_fuse_phoneme'),
            ('OTTC + upstream jsd', 'runexp_ottc_fuse_phoneme_jsd'),
            ('OTTC + standard JSD', 'runexp_ottc_fuse_phoneme_jsd_standard'),
            ('Soft-DTW CE gamma=0.1', 'runexp_softdtw_fuse_phoneme')]
GROUP = 'timit-fused-readme-20261008'


def history_records(history):
    merged = defaultdict(dict)
    summary = {}
    for entry in history:
        row = dict(entry)
        step = int(row.pop('step', 2920))
        epoch = row.pop('epoch', 40)
        if 'train_runtime' in row:
            summary.update(row)
            continue
        item = merged[step]
        item.update({'train/global_step': step, 'train/epoch': epoch})
        for key, value in row.items():
            if isinstance(value, (int, float)):
                name = 'eval/'+key[5:] if key.startswith('eval_') else 'train/'+key
                item[name] = value
        if 'eval_wer' in row:
            item['eval/per_percent'] = 100*row['eval_wer']
    first = {}
    for step in sorted(merged):
        item = merged[step]
        for prefix in ['train', 'eval']:
            key = prefix+'/loss'
            if key in item:
                first.setdefault(prefix, item[key])
                if first[prefix]:
                    item[prefix+'/loss_relative'] = item[key]/first[prefix]
    summary.update({f'baseline/{key}_first_loss': value for key, value in first.items()})
    return [merged[step] for step in sorted(merged)], summary


def upload(api, label, suffix, entity, project):
    output = ROOT/'reproduction/results'/suffix
    metrics = json.loads((output/'metrics.json').read_text())
    model_dir = Path(metrics['checkpoint']).parent
    state_path = model_dir/'trainer_state.json'
    state = json.loads(state_path.read_text())
    assert state['epoch'] == 40 and state['global_step'] == 2920
    assert (output/'TRAIN_COMPLETE.json').exists() and (output/'EVAL_COMPLETE.json').exists()
    config = json.loads((model_dir/'run_config.json').read_text())
    runtime = json.loads((output/'runtime.json').read_text())
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    saved_path = output/'wandb_backfill.json'
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
    if saved and saved.get('complete'):
        assert saved['source_sha256'] == digest
        assert saved['entity'] == entity and saved['project'] == project
        print('ALREADY SYNCED', label, saved['url'], flush=True)
        return saved
    run_id = saved['id'] if saved else wandb.util.generate_id()
    saved = dict(id=run_id, entity=entity, project=project, source_sha256=digest, complete=False,
                 variant=label, origin='backfill')
    saved_path.write_text(json.dumps(saved, indent=2))
    directory = ROOT/'reproduction/results/wandb'
    directory.mkdir(parents=True, exist_ok=True)
    records, summaries = history_records(state['log_history'])
    run = wandb.init(entity=entity, project=project, id=run_id, resume='allow',
                     name=label+' | TIMIT fused | seed 42 | 40 epochs', group=GROUP,
                     job_type='historical-import', mode='online', dir=str(directory),
                     config={**config, 'variant': label, 'logging_origin': 'backfill',
                             'experiment_group': GROUP, 'seed': 42, 'original_logging_steps': 200,
                             'original_runtime': runtime, 'trainer_state_sha256': digest,
                             'softdtw': metrics.get('softdtw'),
                             'loss_scale_note': 'Raw losses have different definitions/reductions; compare per-run trends.'},
                     tags=['backfill', 'seed42', suffix],
                     settings=wandb.Settings(disable_code=True, x_disable_stats=True, init_timeout=45))
    try:
        run.define_metric('train/global_step')
        run.define_metric('train/epoch')
        run.define_metric('*', step_metric='train/global_step', step_sync=True)
        for record in records:
            run.log(record)
        run.summary.update(summaries)
        run.summary.update(benchmark_summary(metrics))
        run.summary.update({'original_final_epoch': 40, 'original_global_step': 2920,
                            'logging_origin': 'backfill', 'source/history_rows': len(records),
                            'source/train_loss_points': sum('train/loss' in r for r in records),
                            'source/eval_loss_points': sum('eval/loss' in r for r in records),
                            'evaluation/audio_only': True, 'evaluation/checkpoint': metrics['checkpoint']})
        url = run.url
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    remote = api.run(f'{entity}/{project}/{run_id}')
    rows = list(remote.scan_history(page_size=200))
    for key in ['train/loss', 'eval/loss', 'eval/per_percent']:
        actual = [(r.get('train/global_step'), r[key]) for r in rows if r.get(key) is not None]
        expected = [(r['train/global_step'], r[key]) for r in records if key in r]
        assert actual == expected, (label, key, len(actual), len(expected))
    saved.update(url=url, complete=True, history_rows=len(records),
                 train_loss_points=sum('train/loss' in r for r in records),
                 eval_loss_points=sum('eval/loss' in r for r in records))
    saved_path.write_text(json.dumps(saved, indent=2))
    print('VERIFIED', label, url, len(records), 'history rows', flush=True)
    return saved


def build_report(entity, project):
    import wandb_workspaces.reports.v2 as wr
    destination = ROOT/'reproduction/results/wandb_report.json'
    previous_url = json.loads(destination.read_text())['url'] if destination.exists() else None
    # Report titles may contain slashes; normalize the slug for SDK URL parsing.
    canonical = f'https://wandb.ai/{entity}/{project}/reports/loss-curves--{previous_url.rsplit("--", 1)[-1]}' if previous_url else None
    report = wr.Report.from_url(canonical) if canonical else wr.Report(
        entity=entity, project=project, title='TIMIT fused phonemes: loss下降与识别及分段对比',
        description='CTC, OTTC CE, upstream jsd, standard JSD and Soft-DTW; verified historical logs.')
    report.title = 'TIMIT fused phonemes: loss下降与识别及分段对比'
    runset = wr.Runset(entity=entity, project=project, name='Five completed 40-epoch runs',
                      filters=f'Config("experiment_group") = "{GROUP}"')
    def plot(title, ys, **kwargs):
        return wr.LinePlot(title=title, x='train/epoch', y=ys, aggregate=False,
                           smoothing_factor=0, smoothing_type='none', max_runs_to_show=10,
                           title_x='Epoch', **kwargs)
    report.blocks = [wr.MarkdownBlock(text='''
这五组均为官方 LV60 底座、TIMIT fused-phoneme、seed 42、40 epochs / 2920 steps。
图中是原始 Trainer 日志的事后补录：训练 loss 每 200 步记录，测试 loss 每轮记录。
横轴为 epoch，可在图中缩放、悬停并选择 run。没有补造缺失的逐步 loss。

各 loss 的定义与 reduction 不同，不能直接用数值高低判断模型优劣。
相对 loss 以各自第一条实际记录为 1，仅用于观察下降趋势。
原仓库 `jsd` 是反向 KL 到混合分布的代价，与标准 JSD 分开记录。
'''), wr.PanelGrid(runsets=[runset], panels=[
        plot('训练 loss 相对第一条记录', ['train/loss_relative']),
        plot('测试 loss 相对第一轮', ['eval/loss_relative']),
        plot('音素错误率 PER (%)', ['eval/per_percent']),
        plot('学习率', ['train/learning_rate'])])]
    for label, suffix in VARIANTS:
        one = wr.Runset(entity=entity, project=project, name=label,
                        filters=f'Config("experiment_group") = "{GROUP}" and Config("suffix") = "{suffix}"')
        report.blocks.extend([wr.MarkdownBlock(text=f'### {label}：原始 loss'),
                              wr.PanelGrid(runsets=[one], panels=[plot(label+' train / eval loss', ['train/loss', 'eval/loss'])])])
    soft = wr.Runset(entity=entity, project=project, name='Soft-DTW late training',
                     filters=f'Config("experiment_group") = "{GROUP}" and Config("loss_type") = "softdtw"')
    report.blocks.extend([wr.MarkdownBlock(text='''
### Soft-DTW 后期观察
第 20/30/40 轮测试 loss 为 109.06 / 107.17 / 103.96，PER 为 77.00% / 74.84% / 74.54%。
loss 仍缓慢下降，识别已经接近平台，学习率在第 40 轮降到零，尚不能确认充分收敛。
'''), wr.PanelGrid(runsets=[soft], panels=[
        plot('Soft-DTW 第20–40轮 loss', ['train/loss', 'eval/loss'], range_x=(20, 40), range_y=(100, 115)),
        plot('Soft-DTW 第20–40轮 PER', ['eval/per_percent'], range_x=(20, 40), range_y=(72, 79))])])
    report.save()
    info = dict(url=report.url, entity=entity, project=project,
                project_url=f'https://wandb.ai/{entity}/{project}', group=GROUP)
    destination.write_text(json.dumps(info, indent=2))
    print('REPORT', report.url, flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    settings = defaults()
    parser.add_argument('--project', default=settings['project'])
    parser.add_argument('--entity', default=settings['entity'])
    args = parser.parse_args()
    api = wandb.Api(timeout=45)
    entity = args.entity or api.default_entity
    saved = {}
    for label, suffix in VARIANTS:
        saved[suffix] = upload(api, label, suffix, entity, args.project)
    report = build_report(entity, args.project)
    (ROOT/'reproduction/results/WANDB_COMPLETE.json').write_text(json.dumps(dict(runs=saved, report=report), indent=2))


if __name__ == '__main__':
    main()
