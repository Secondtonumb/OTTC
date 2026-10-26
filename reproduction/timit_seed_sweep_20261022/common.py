"""Matched single-GPU TIMIT recipes; only objective and random seed vary."""
from pathlib import Path
import json

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, build_model, load_data, Collator, parameter_manifest,
    sha, write, phones_from_ids, normalize, scorer,
)
from reproduction.timit_divergences_20261022.common import snapshot_sources as snapshot_internal

HERE = Path(__file__).resolve().parent
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')


def source_hashes():
    paths = [HERE / name for name in ('common.py', 'train.py', 'profile_recipe.py', 'evaluate.py', 'pfer.py', 'run.pbs')]
    paths += [ROOT.parent / 'fa-bench/analysis/panphon_pfer/distance.py',
              ROOT.parent / 'fa-bench/analysis/panphon_pfer/feature_distance.so',
              ROOT.parent / 'seamless_communication/outputs/panphon_libri100/panphon_table.json']
    return dict(base.source_hashes(), **{str(path): sha(path) for path in paths})


def recipe(name):
    cost, seed_text = name.split('__seed')
    seed = int(seed_text)
    if cost not in COSTS or seed < 0:
        raise ValueError(name)
    baseline_cost = 'jsd_upstream' if cost in ('ag', 'jdiv') else cost
    baseline = ROOT / 'reproduction/timit_ssl_native10_20261019/models' / f'wav2vec2__timit__10ms__mono__{baseline_cost}'
    config = base.recipe(f'wav2vec2__timit__10ms__mono__{baseline_cost}')
    config.update(name=name, experiment_family=HERE.name, cost=cost, seed=seed,
                  bins_metric={'ce': 'crossentropy', 'jsd_standard': 'jsd_standard', 'ag': 'ag', 'jdiv': 'jdiv'}[cost],
                  evaluation_uploads=False, temporal_smoothing_weight=0.,
                  baseline=str(baseline), source_sha256=source_hashes(),
                  checkpoint_policy='best three dev-PER weights only; one latest full resume saved every10 epochs or interruption',
                  comparison_contract='FP32 best-dev single checkpoint against historical FP32 best-dev; final separately',
                  objective={'ce': 'original OT-weighted cross entropy', 'jsd_standard': 'standard Jensen-Shannon',
                             'ag': 'exact legacy upstream jsd arithmetic-geometric divergence',
                             'jdiv': 'KL(p||q_eps)+KL(q_eps||p), no half factor'}[cost])
    previous = json.loads((baseline / 'run_config.json').read_text())
    fields = ('epochs', 'global_batch', 'learning_rate', 'warmup_steps', 'weight_decay', 'backbone',
              'frame_ms', 'head', 'forward_type', 'native_conv_stride', 'target_mass',
              'freeze_cnn_first_six', 'train_cnn_last_layer', 'train_transformer', 'train_all_task_heads')
    assert all(config[key] == previous[key] for key in fields)
    return config


def atomic_weights(model, path, metadata):
    import torch
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    base.atomic_weights(model, path, metadata)


def snapshot_sources(output, hashes):
    internal = {source: digest for source, digest in hashes.items() if Path(source).is_relative_to(ROOT)}
    snapshot_internal(output, internal)
    for source, digest in hashes.items():
        if source in internal:
            continue
        path = Path(source)
        assert sha(path) == digest
        target = output / 'source_snapshot/workspace' / path.relative_to(ROOT.parent)
        if target.exists():
            assert sha(target) == digest
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_bytes(path.read_bytes())
            temporary.replace(target)


def retain_best(output, model, value, epoch, step, config):
    index = output / 'BEST_THREE.json'
    previous = json.loads(index.read_text())['checkpoints'] if index.exists() else []
    if any(item['step'] == step for item in previous):
        return
    candidate = dict(PER=value, epoch=epoch, step=step, path=str(output / 'best_weights' / f'step-{step:06d}.pt'))
    keep = sorted(previous + [candidate], key=lambda item: (item['PER'], item['step']))[:3]
    if candidate not in keep:
        return
    atomic_weights(model, Path(candidate['path']), dict(epoch=epoch, step=step, recipe=config['name'],
                   dev_PER=value, seed=config['seed'], model_config=model.config.to_dict(), source_sha256=config['source_sha256']))
    write(index, dict(criterion='lowest development PER; test unused', checkpoints=keep))
    link = output / '.best_dev.tmp'
    link.unlink(missing_ok=True)
    link.symlink_to(Path(keep[0]['path']).relative_to(output))
    link.replace(output / 'best_dev.pt')
    write(output / 'BEST_DEV.json', dict(**keep[0], selection='development PER every epoch; test unused'))
    for item in previous:
        if item not in keep:
            path = Path(item['path'])
            assert path.parent == output / 'best_weights'
            path.unlink()
