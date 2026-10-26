"""Fresh SSL, full40 Buckeye selection, independent Buckeye/TIMIT evaluation."""
import json
from pathlib import Path

from reproduction.buckeye_dataset import load_buckeye_rows, resolve_buckeye
from reproduction.timit_ssl_native10_20261019 import common as base

HERE = Path(__file__).resolve().parent
ROOT = base.ROOT
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')
SEEDS = (42, 43, 44)


def checked_rows():
    data_root, complete = resolve_buckeye()
    assert complete['source_speakers'] == 40 and complete['source_tracks'] == 255
    assert complete['split_speaker_and_track_disjoint']
    rows, contracts = {}, {}
    seen_speakers, seen_ids = set(), set()
    for split in ('train', 'dev', 'test'):
        current, vocab, contract = load_buckeye_rows(split, training_speakers=seen_speakers)
        identifiers = {row['utt'] for row in current}
        assert len(identifiers) == len(current) and not seen_ids.intersection(identifiers)
        seen_ids.update(identifiers)
        seen_speakers.update(row['speaker'] for row in current)
        rows[split], contracts[split] = current, contract
    assert len(seen_speakers) == 40
    timit_root = ROOT / 'reproduction/multihead_timit/data'
    timit_vocab = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json'
    assert vocab == json.loads(timit_vocab.read_text()), 'TIMIT phone IDs differ'
    for split, filename, count in (('timit_dev', 'train-dev.jsonl', 400), ('timit_test', 'test.jsonl', 192)):
        path = timit_root / filename
        with path.open() as stream:
            originals = [json.loads(line) for line in stream]
        assert len(originals) == len({row['utt'] for row in originals}) == count
        rows[split] = [dict(utt=row['utt'], wav=row['wav'], duration=row['duration'],
                            reference=row['references']['ifmdd_arpabet'],
                            training_target_ids=row['targets']['ottc_arpabet']) for row in originals]
        contracts[split] = dict(manifest=str(path), sha256=base.sha(path), utterances=count)
    return rows, vocab, contracts, data_root


def source_hashes():
    from reproduction.libri100_timit_seed_sweep_20261022.common import source_hashes as parent_hashes
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.pbs')) + list(HERE.glob('*.sh'))
    paths += [ROOT / 'reproduction/buckeye_dataset.py', ROOT / 'reproduction/dataset_defaults.json']
    paths += [ROOT / 'reproduction/libri100_timit_seed_sweep_20261022/train.py',
              ROOT / 'reproduction/buckeye_seed_sweep_20261022/profile_recipe.py',
              ROOT / 'reproduction/timit_divergences_20261022/profile_recipe.py',
              ROOT / 'reproduction/timit_seed_sweep_20261022/common.py',
              ROOT / 'reproduction/timit_seed_sweep_20261022/pfer.py',
              ROOT / 'reproduction/libri960_continue5_20261017/environment.sh']
    paths += [ROOT / 'reproduction/timit_seed_sweep_20261022/report.py']
    paths += [ROOT / 'reproduction/checkpoint_retention.py']
    return dict(parent_hashes(), **{str(path): base.sha(path) for path in paths})


def recipe(name, contracts=None):
    cost, seed_text = name.split('__seed')
    seed = int(seed_text)
    if cost not in COSTS or seed not in SEEDS:
        raise ValueError(name)
    config = base.recipe('wav2vec2__timit__10ms__mono__' + ('jsd_upstream' if cost in ('ag', 'jdiv') else cost))
    if contracts is None:
        _, _, contracts, _ = checked_rows()
    config.update(name=name, experiment_family=HERE.name, corpus='buckeye', cost=cost, seed=seed,
                  bins_metric={'ce': 'crossentropy', 'jsd_standard': 'jsd_standard', 'ag': 'ag', 'jdiv': 'jdiv'}[cost],
                  epochs=40, global_batch=64, evaluation_uploads=False, temporal_smoothing_weight=0.,
                  expected_data=contracts, selection_corpus='Buckeye full40 dev only',
                  initialization='original wav2vec2-large-lv60 SSL; fresh random task heads; no OTTC checkpoint',
                  checkpoint_policy='dev-best three FP32 weights only; latest one resumable state; best-dev reuses retained weights',
                  evaluation_candidates=['best_dev', 'best3'],
                  selection_rule='Buckeye dev PER only; Buckeye test and all TIMIT splits never select',
                  source_sha256=source_hashes())
    return config


class AudioDataset:
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor
        self.lengths = [round(row['duration'] * 16000) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import soundfile as sf
        row = self.rows[index]
        audio, rate = sf.read(row['wav'], dtype='float32')
        assert rate == 16000 and audio.ndim == 1 and len(audio) == self.lengths[index]
        values = self.processor(audio, sampling_rate=rate).input_values[0]
        return dict(input_values=values, labels=row['training_target_ids'], length=len(audio))