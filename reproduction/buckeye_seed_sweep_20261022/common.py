"""Fresh-SSL Buckeye native10ms ablations with independent batch64 optimizers."""
import json
from pathlib import Path

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, build_model, Collator, parameter_manifest,
    sha, write, phones_from_ids, normalize, scorer,
)
from reproduction.timit_seed_sweep_20261022.common import (
    atomic_weights, snapshot_sources, retain_best,
)

HERE = Path(__file__).resolve().parent
DATA = ROOT / 'reproduction/buckeye_singlehead_20261014/data'
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')
SEEDS = (42, 43, 44)


def source_hashes():
    paths = [HERE / name for name in ('common.py', 'train.py', 'profile_recipe.py', 'worker.sh', 'run.pbs')]
    paths += [ROOT / 'reproduction/timit_seed_sweep_20261022' / name for name in ('common.py', 'evaluate.py', 'pfer.py')]
    paths += [ROOT / 'reproduction/timit_divergences_20261022' / name for name in ('common.py', 'profile_recipe.py')]
    paths += [ROOT.parent / 'fa-bench/analysis/panphon_pfer/distance.py',
              ROOT.parent / 'fa-bench/analysis/panphon_pfer/feature_distance.so',
              ROOT.parent / 'seamless_communication/outputs/panphon_libri100/panphon_table.json']
    return dict(base.source_hashes(), **{str(path): sha(path) for path in paths})


def expected_data():
    audit = json.loads((DATA / 'audit.json').read_text())
    assert audit['speaker_utterance_source_disjointness_verified']
    hashes = {split: sha(DATA / f'{split}.jsonl') for split in ('train', 'dev', 'test')}
    assert all(hashes[split] == audit['splits'][split]['prepared_manifest_sha256'] for split in hashes)
    return dict(manifests_sha256=hashes, vocab_sha256=sha(DATA / 'vocab.json'),
                counts={split: audit['splits'][split]['utterances'] for split in hashes},
                references='human PHN for TIMIT; original human partial19 Buckeye; existing automatic Libri boundaries, source model unverified')


def recipe(name):
    cost, seed_text = name.split('__seed')
    seed = int(seed_text)
    if cost not in COSTS or seed not in SEEDS:
        raise ValueError(name)
    baseline_cost = 'jsd_upstream' if cost in ('ag', 'jdiv') else cost
    config = base.recipe(f'wav2vec2__timit__10ms__mono__{baseline_cost}')
    config.update(name=name, experiment_family=HERE.name, corpus='buckeye', cost=cost, seed=seed,
                  bins_metric={'ce': 'crossentropy', 'jsd_standard': 'jsd_standard', 'ag': 'ag', 'jdiv': 'jdiv'}[cost],
                  evaluation_uploads=False, temporal_smoothing_weight=0., expected_data=expected_data(),
                  dataset_scope='existing original-human partial19; speaker-disjoint 15/2/2; not full 40-speaker Buckeye',
                  launcher_contract='12 nodes, one independent experiment/GPU/node; no cross-objective gradient synchronization',
                  checkpoint_policy='three lowest dev-PER weights and one latest full resume every10 epochs/interruption',
                  comparison_contract='FP32 single best-dev checkpoint; identical splits, scorer and seeds across four objectives',
                  source_sha256=source_hashes())
    return config


def load_data(config):
    data, rows, vocab, processor, audit = base.load_data(config)
    assert audit == config['expected_data']
    seen = set()
    speakers = set()
    for split in ('train', 'dev', 'test'):
        identifiers = {row['utt'] for row in rows[split]}
        split_speakers = {row['speaker'] for row in rows[split]}
        assert len(identifiers) == len(rows[split])
        assert not seen.intersection(identifiers)
        assert not speakers.intersection(split_speakers)
        seen.update(identifiers)
        speakers.update(split_speakers)
        if split in data:
            assert len(data[split]) == len(rows[split])
            assert data[split]['labels'] == [row['training_target_ids'] for row in rows[split]]
    return data, rows, vocab, processor, audit
