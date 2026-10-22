"""Fresh-SSL native10ms TIMIT divergence recipes, isolated from old resumes."""
from pathlib import Path

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, Collator, atomic_weights, build_model, load_data,
    normalize, parameter_manifest, phones_from_ids, scorer, sha, write,
)

HERE = Path(__file__).resolve().parent
COSTS = ('ag', 'kl', 'kl_reverse', 'jdiv', 'jsd_standard')


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh'))
    # The inherited factory and scoring/data helpers are part of the contract.
    return dict(base.source_hashes(), **{str(p): sha(p) for p in paths})


def recipe(cost, epochs=40, batch_size=64, eval_every=5):
    if cost not in COSTS:
        raise ValueError(f'Unknown divergence {cost!r}')
    if epochs < 3 or batch_size < 1 or eval_every < 1:
        raise ValueError('epochs must be >=3; batch_size/eval_every must be positive')
    config = base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    config.update(
        name=f'wav2vec2__timit__10ms__mono__{cost}__e{epochs}_b{batch_size}',
        experiment_family=HERE.name, cost=cost, bins_metric=cost,
        epochs=epochs, global_batch=batch_size, evaluation_uploads=False,
        temporal_smoothing_weight=0.,
        eval_every=eval_every, average_epochs=list(range(epochs-2, epochs+1)),
        objective='OT-weighted class-summed divergence; utterance losses summed, divided by actual logical batch size for backward',
        divergence_definition={
            'ag': 'legacy upstream AG including original unnormalized target epsilon perturbation',
            'kl': 'KL(p||q_eps)', 'kl_reverse': 'KL(q_eps||p)',
            'jdiv': 'KL(p||q_eps)+KL(q_eps||p), no half factor',
            'jsd_standard': '.5 KL(p||m)+.5 KL(q||m); m=(p+q)/2',
        }[cost],
        target_regularization='normalize(clamp(q,min=1e-12))' if cost in ('kl', 'kl_reverse', 'jdiv') else 'original selected branch',
        checkpoint_policy='user update: save/evaluate every5 epochs plus final; save final three epoch weights; average final three; full latest resume on scheduled saves or interruption',
        source_sha256=source_hashes(),
    )
    return config


def snapshot_sources(output, hashes):
    """Freeze source evidence without touching historical source snapshots."""
    for source, expected in hashes.items():
        path = Path(source)
        assert sha(path) == expected, f'Source changed: {path}'
        target = output / 'source_snapshot' / path.relative_to(ROOT)
        if target.exists():
            assert sha(target) == expected, f'Snapshot changed: {target}'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_bytes(path.read_bytes())
            temporary.replace(target)


def inspect_inputs():
    """Read-only preflight usable on the login node, without loading SSL weights."""
    import json
    manifest_dir = ROOT / 'reproduction/multihead_timit/data'
    report = {}
    for split, filename, expected in (
        ('train', 'train-train.jsonl', 4620), ('dev', 'train-dev.jsonl', 400),
        ('test', 'test.jsonl', 192),
    ):
        path = manifest_dir / filename
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == len({row['utt'] for row in rows}) == expected
        missing = [row['wav'] for row in rows if not Path(row['wav']).is_file()]
        assert not missing, f'Missing audio: {missing[:3]}'
        report[split] = dict(count=len(rows), manifest=str(path), sha256=sha(path))
    assert (SSL / 'pytorch_model.bin').is_file()
    report['ssl'] = dict(path=str(SSL), expected_sha256=SSL_SHA,
                         hash_verified=False, note='Full SSL identity is checked by the GPU profile')
    return report
