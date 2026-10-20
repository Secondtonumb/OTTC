"""W&B run identity, live Trainer logging and final benchmark summaries."""
import json
import os
from pathlib import Path

from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT/'reproduction/wandb_config.json'


def defaults():
    values = dict(project='ottc-timit-reproduction', entity=None, mode='online')
    if DEFAULTS_PATH.exists():
        values.update(json.loads(DEFAULTS_PATH.read_text()))
    return values


def start_training_run(config, model_output_dir):
    """Called only on process zero; preserve identity on checkpoint resume."""
    import wandb
    output = Path(model_output_dir)
    identity = output/'wandb_run.json'
    previous = json.loads(identity.read_text()) if identity.exists() else {}
    resume = bool(config['continue_if_exist'] and any(output.glob('checkpoint-*')))
    directory = ROOT/'reproduction/results/wandb'
    directory.mkdir(parents=True, exist_ok=True)
    mode = config['wandb_mode']
    project, entity = config['wandb_project'], config['wandb_entity']
    if resume and previous:
        if previous['project'] != project or previous['entity'] != entity:
            raise ValueError('Resume must use the recorded W&B project/entity')
    kwargs = dict(project=project, entity=entity, name=config['suffix'],
                  group=os.environ.get('WANDB_RUN_GROUP', config['dataset']),
                  job_type='training', config={**config, 'logging_origin': 'live'},
                  tags=[config['loss_type'], 'live'], dir=str(directory), mode=mode,
                  settings=wandb.Settings(init_timeout=45, disable_code=True))
    if resume and previous:
        kwargs.update(id=previous['id'], resume='allow')
    try:
        run = wandb.init(**kwargs)
    except wandb.errors.CommError:
        if mode != 'online':
            raise
        # Keep local W&B records when compute-node networking is unavailable.
        kwargs.update(mode='offline')
        kwargs.pop('resume', None)
        run = wandb.init(**kwargs)
        mode = 'offline'
        print('W&B connection unavailable; recording offline for later sync.', flush=True)
    run.define_metric('train/global_step')
    run.define_metric('train/epoch')
    run.define_metric('*', step_metric='train/global_step', step_sync=True)
    identity.write_text(json.dumps(dict(id=run.id, entity=run.entity or entity,
                                       project=run.project, url=run.url if mode == 'online' else None,
                                       mode=mode, origin='live'), indent=2))
    run.config.update(dict(dataset_sizes=json.loads((output/'dataset_sizes.json').read_text())
                           if (output/'dataset_sizes.json').exists() else None), allow_val_change=True)
    return run


class LossCurveCallback(TrainerCallback):
    """Add percentage PER and per-run relative losses alongside HF raw metrics."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        import wandb
        run = wandb.run
        if run is None:
            return
        values = {'train/global_step': state.global_step, 'train/epoch': state.epoch}
        for source, target in [('loss', 'train'), ('eval_loss', 'eval')]:
            if source in logs:
                base_key = f'baseline/{target}_first_loss'
                base = run.summary.get(base_key)
                if base is None:
                    base = float(logs[source])
                    run.summary[base_key] = base
                if base:
                    values[f'{target}/loss_relative'] = float(logs[source])/base
        if 'eval_wer' in logs:
            values['eval/per_percent'] = 100*logs['eval_wer']
        run.log(values)


def finish_training_run(exit_code=0):
    import wandb
    if wandb.run is not None:
        wandb.finish(exit_code=exit_code)


def benchmark_summary(results):
    values = {}
    for split, metrics in results['splits'].items():
        prefix = f'benchmark/{split}'
        values[f'{prefix}/raw_PER_percent'] = 100*metrics['raw_PER']
        values[f'{prefix}/nosil_PER_percent'] = 100*metrics['normalized_nosil_PER']
        values[f'{prefix}/interval_overlap_percent'] = 100*metrics['matched_interval_overlap_ratio']
        for tolerance, boundary in metrics['boundaries'].items():
            for key in ['precision', 'recall', 'f1', 'r_value']:
                values[f'{prefix}/boundary_{key}_{tolerance}ms_percent'] = 100*boundary[key]
    return values


def publish_benchmark(results, model_dir):
    """Append summaries to the live training run; legacy runs are backfilled separately."""
    from reproduction.wandb_policy import evaluation_uploads_enabled
    if not evaluation_uploads_enabled():
        return
    identity = Path(model_dir)/'wandb_run.json'
    if not identity.exists():
        return
    saved = json.loads(identity.read_text())
    if saved['origin'] != 'live':
        return
    import wandb
    directory = ROOT/'reproduction/results/wandb'
    directory.mkdir(parents=True, exist_ok=True)
    with wandb.init(project=saved['project'], entity=saved['entity'], id=saved['id'],
                    resume='allow' if saved['mode'] == 'online' else None,
                    mode=saved['mode'], dir=str(directory), job_type='evaluation',
                    settings=wandb.Settings(init_timeout=45, disable_code=True)) as run:
        run.summary.update(benchmark_summary(results))
        run.summary['evaluation/audio_only'] = True
        run.summary['evaluation/checkpoint'] = results['checkpoint']
