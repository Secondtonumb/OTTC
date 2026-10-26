"""Libri100-only training and selection; fixed TIMIT evaluation, never TIMIT training."""
from functools import lru_cache
import json
from pathlib import Path

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, build_model, Collator, parameter_manifest,
    sha, write, phones_from_ids, normalize, scorer,
)
from reproduction.timit_seed_sweep_20261022.common import atomic_weights, snapshot_sources, retain_best

HERE = Path(__file__).resolve().parent
DATA = ROOT / 'reproduction/libri100_singlehead_20261014/data'
TIMIT = ROOT / 'reproduction/multihead_timit/data'
TIMIT_PROCESSOR = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor'
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')
SEEDS = (42, 43, 44)


def source_hashes():
    paths = [HERE / name for name in ('common.py', 'train.py', 'profile_recipe.py', 'evaluate.py', 'worker.sh', 'run.pbs')]
    paths += [ROOT / 'reproduction/timit_seed_sweep_20261022' / name for name in ('common.py', 'pfer.py')]
    paths += [ROOT / 'reproduction/timit_divergences_20261022' / name for name in ('common.py', 'profile_recipe.py')]
    paths += [ROOT.parent / 'fa-bench/analysis/panphon_pfer/distance.py',
              ROOT.parent / 'fa-bench/analysis/panphon_pfer/feature_distance.so',
              ROOT.parent / 'seamless_communication/outputs/panphon_libri100/panphon_table.json']
    return dict(base.source_hashes(), **{str(path): sha(path) for path in paths})


@lru_cache(maxsize=1)
def expected_data():
    audit = json.loads((DATA / 'audit.json').read_text())
    assert audit['speaker_utterance_source_disjointness_verified']
    expected = {}
    for split, count in (('train', 28538), ('dev', 5567)):
        path = DATA / f'{split}.jsonl'
        digest = sha(path)
        assert audit['splits'][split]['utterances'] == count
        assert digest == audit['splits'][split]['prepared_manifest_sha256']
        expected[split] = dict(corpus='libri100', count=count, manifest=str(path), sha256=digest)
    for split, filename, count in (('timit_dev', 'train-dev.jsonl', 400), ('timit_test', 'test.jsonl', 192)):
        path = TIMIT / filename
        expected[split] = dict(corpus='timit', count=count, manifest=str(path), sha256=sha(path))
    vocab = json.loads((DATA / 'vocab.json').read_text())
    assert vocab == json.loads((TIMIT_PROCESSOR / 'vocab.json').read_text()), 'Phone-ID mappings differ'
    expected['vocab_sha256'] = sha(DATA / 'vocab.json')
    expected['selection'] = 'Libri dev-clean+dev-other canonical sequence PER; TIMIT dev/test never select checkpoint'
    return expected


def recipe(name):
    cost, seed_text = name.split('__seed')
    seed = int(seed_text)
    if cost not in COSTS or seed not in SEEDS:
        raise ValueError(name)
    baseline_cost = 'jsd_upstream' if cost in ('ag', 'jdiv') else cost
    config = base.recipe(f'wav2vec2__timit__10ms__mono__{baseline_cost}')
    config.update(name=name, experiment_family=HERE.name, corpus='libri100', cost=cost, seed=seed,
                  bins_metric={'ce': 'crossentropy', 'jsd_standard': 'jsd_standard', 'ag': 'ag', 'jdiv': 'jdiv'}[cost],
                  evaluation_uploads=False, temporal_smoothing_weight=0., expected_data=expected_data(),
                  dataset_scope='complete existing train-clean-100:28538 utterances,100.5865h; no TIMIT training',
                  selection_corpus='LibriSpeech dev-clean+dev-other', evaluation_corpus='TIMIT dev400/test192',
                  selection_reference='mapped canonical phone sequences in Libri dev cache; no reference timing used',
                  launcher_contract='12 nodes, one independent experiment/GPU/node; no cross-objective gradient synchronization',
                  checkpoint_policy='three lowest Libri dev-PER weights; one latest full resume every10 epochs/interruption',
                  comparison_contract='FP32 single best-Libri-dev checkpoint on fixed TIMIT dev/test; no checkpoint averages',
                  objective={'ce': 'original OT-weighted cross entropy', 'jsd_standard': 'standard Jensen-Shannon',
                             'ag': 'legacy upstream arithmetic-geometric divergence including target epsilon',
                             'jdiv': 'KL(p||q_eps)+KL(q_eps||p), no half factor'}[cost],
                  source_sha256=source_hashes())
    return config


def load_data(config):
    import datasets
    from transformers import Wav2Vec2Processor
    audit = expected_data()
    assert audit == config['expected_data']
    vocab = json.loads((DATA / 'vocab.json').read_text())
    assert set(vocab.values()) == set(range(42)) and vocab['<pad>'] == 0 and vocab['<unk>'] == 41
    processor = Wav2Vec2Processor.from_pretrained(DATA / 'processor')
    data, rows = {}, {}
    seen = set()
    for split in ('train', 'dev'):
        path = Path(audit[split]['manifest'])
        assert sha(path) == audit[split]['sha256']
        originals = [json.loads(line) for line in path.open()]
        identifiers = {row['utt'] for row in originals}
        assert len(identifiers) == len(originals) == audit[split]['count']
        assert not seen.intersection(identifiers)
        seen.update(identifiers)
        data[split] = datasets.load_from_disk(DATA / f'cache_{split}')
        assert len(data[split]) == len(originals)
        assert set(data[split].column_names) == {'input_values', 'labels', 'length'}
        assert data[split]['labels'] == [row['training_target_ids'] for row in originals]
        assert data[split]['length'] == [round(row['duration'] * 16000) for row in originals]
        rows[split] = [dict(utt=row['utt'], duration=row['duration']) for row in originals]
    for split in ('timit_dev', 'timit_test'):
        path = Path(audit[split]['manifest'])
        assert sha(path) == audit[split]['sha256']
        originals = [json.loads(line) for line in path.open()]
        assert len(originals) == len({row['utt'] for row in originals}) == audit[split]['count']
        rows[split] = [dict(utt=row['utt'], wav=row['wav'], duration=row['duration'],
                            reference=row['references']['ifmdd_arpabet'],
                            training_target_ids=row['targets']['ottc_arpabet']) for row in originals]
    assert not {row['utt'] for row in rows['timit_dev']}.intersection(row['utt'] for row in rows['timit_test'])
    return data, rows, vocab, processor, audit
