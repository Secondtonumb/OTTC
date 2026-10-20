"""Publish numeric full-set benchmark summaries only; no audio/text artifacts."""
import json
import os

from reproduction.ottc_alpha_full_20261017.prepare import ROOT, HERE, write


def main():
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    result = json.loads((HERE / 'RESULTS.json').read_text())
    contract = json.loads((HERE / 'DECODE_CONTRACT.json').read_text())
    assert result['status'] == 'complete' and result['utterances'] == 7590
    os.environ.setdefault('WANDB_CACHE_DIR', str(HERE / 'wandb_cache'))
    identities = []
    for model in result['models']:
        dataset = model['dataset']
        directory = HERE / 'wandb_runs' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        identity = directory / 'wandb_run.json'
        saved = json.loads(identity.read_text()) if identity.exists() else {}
        if saved.get('finished') and saved.get('mode') == 'online':
            identities.append(saved)
            continue
        run_id = saved.get('id') or wandb.util.generate_id()
        entity = os.environ.get('WANDB_ENTITY', defaults['entity'])
        project = os.environ.get('WANDB_PROJECT', defaults['project'])
        mode = os.environ.get('WANDB_MODE', defaults.get('mode', 'online'))
        write(identity, dict(id=run_id, dataset=dataset, entity=entity, project=project, mode=mode, finished=False))
        options = dict(id=run_id, entity=entity, project=project,
            name=dataset + ' | full test | upstream JSD alpha decoding',
            group='ottc_alpha_full_20261017', job_type='segmentation-evaluation', dir=str(directory),
            tags=['full-test-set', 'fresh-inference', 'audio-only', 'original-upstream-JSD'],
            config=dict(result['protocol'], dataset=dataset, utterances=model['utterances'],
                evaluation_only=True, checkpoint_sha256=model['checkpoint_sha256'],
                original_epoch=model['original_epoch'], original_step=model['original_step'],
                precision='FP32', batch_size=1, source_decode_sha256=contract['source_sha256'][str(HERE / 'decode.py')]),
            settings=wandb.Settings(init_timeout=25, disable_code=True, x_disable_stats=True))
        try:
            run = init_evaluation(mode=mode, resume='allow' if mode == 'online' else None, **options)
        except wandb.errors.Error:
            mode = 'offline'
            run = init_evaluation(mode=mode, **options)
        values = dict(utterances=model['utterances'], optimizer_updates=0,
            original_training_epoch=model['original_epoch'], original_training_step=model['original_step'],
            peak_allocated_GiB=model['peak_allocated_GiB'], inference_seconds=model['seconds'],
            checkpoint_unchanged=model['checkpoint_unchanged'], recognition_sequence_unchanged=True)
        for split, metrics in result['splits'].items():
            if not (split.startswith(dataset) or (dataset == 'libri100' and split.startswith('libri_'))):
                continue
            values[split + '/utterances'] = metrics['utterances']
            values[split + '/nosil_PER_percent'] = metrics['normalized_nosil_PER_percent']
            for key, value in metrics['diagnostics'].items():
                values[split + '/diagnostic/' + key] = value
            for method, by_ms in metrics['methods'].items():
                for ms, m in by_ms.items():
                    for key in ['matches', 'predicted_boundaries', 'reference_boundaries',
                                'precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct']:
                        values[f'{split}/{method}/{key}_{ms}ms'] = m[key]
        run.log(values)
        run.summary.update(values)
        value = dict(id=run.id, dataset=dataset, entity=entity, project=project, mode=mode,
            url=run.url if mode == 'online' else None, finished=True,
            uploaded='numeric configuration and benchmark summaries only')
        run.finish()
        write(identity, value)
        identities.append(value)
    write(HERE / 'WANDB_RUNS.json', identities)
    print(json.dumps(identities), flush=True)


if __name__ == '__main__':
    main()
