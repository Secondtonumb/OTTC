"""One Buckeye full40 frame-AG run; immutable historical data and model helpers."""
import csv
import json
from pathlib import Path
import statistics

from transformers import Wav2Vec2Processor
from reproduction.buckeye_full40_seed_sweep_20261025 import common as buckeye
from reproduction.timit_frame_divergences_20261026 import common as parent
from reproduction.timit_frame_ce_20261026.common import (
    ROOT, SSL, SSL_SHA, sha, write, phones_from_ids, normalize, scorer,
    parameter_manifest, Collator, project, frame_count, atomic_weights,
    snapshot_sources, retain_best,
)

HERE = Path(__file__).resolve().parent
NAME = 'frame_ag__seed42'


class AudioDataset(buckeye.AudioDataset):
    def __getitem__(self, index):
        example = super().__getitem__(index)
        example['wav'] = self.rows[index]['wav']
        return example


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + [HERE / 'run.pbs']
    paths += [ROOT / 'reproduction/buckeye_dataset.py',
              ROOT / 'reproduction/dataset_defaults.json',
              ROOT / 'reproduction/buckeye_full40_seed_sweep_20261025/common.py',
              ROOT / 'reproduction/buckeye_full40_exactfa_20261026/exact.py']
    return dict(parent.source_hashes(), **{str(p): sha(p) for p in paths})


def choice():
    source = ROOT / 'reproduction/timit_frame_divergences_20261026/RESULTS.csv'
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    means = {}
    for method in ('frame_jsd_standard', 'frame_jdiv', 'frame_ag'):
        selected = [r for r in rows if r['method'] == method and
                    r['candidate'] == 'best_dev' and r['split'] == 'dev']
        assert sorted(int(r['seed']) for r in selected) == [42, 43, 44]
        means[method] = statistics.mean(float(r['PER']) for r in selected)
    assert min(means, key=means.get) == 'frame_ag'
    return dict(criterion='lowest three-seed TIMIT dev PER among frame divergences; no test selection',
                source=str(source), sha256=sha(source), dev_PER_means=means,
                seed_policy='fixed seed42; one independent run, no seed search')


def recipe(name):
    assert name == NAME
    config = parent.recipe(name)
    config.pop('baseline', None)
    config.update(experiment_family=HERE.name, corpus='buckeye', seed=42,
        initialization='original wav2vec2-large-lv60 SSL encoder; random phone head; no supervised parent',
        frame_supervision='Buckeye full40 reference intervals; original timestamps; gaps ignored; no artificial edge sil targets',
        frame_center_samples='199.5 + 160 * frame_index at 16kHz (400-sample receptive field)',
        parallelism='one node / one GH200 / world_size=1; seed42 only',
        divergence_choice=choice(), evaluation_uploads=False,
        checkpoint_policy='atomic FP32 weights every full epoch, retain all40; best-dev symlink; latest full resume; final5/final10 FP32 averages',
        selection_rule='Buckeye dev PER only; primary single best-dev; averages selected on Buckeye dev; tests and TIMIT never select',
        FA_search='unpruned exact standard CTC Viterbi; no tail exclusions',
        source_sha256=source_hashes())
    return config


def load_data(config):
    rows, vocab, contracts, data_root = buckeye.checked_rows()
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    timelines = {}
    coverage = {}
    # Full40 mapped reference intervals supply the training labels. Historical
    # training token sequences include edge sil but supply no invented times.
    # TIMIT is evaluation-only; use its actual PHN intervals for loss diagnostics.
    timit = {}
    for split, filename in [('timit_dev', 'train-dev'), ('timit_test', 'test')]:
        path = ROOT / 'reproduction/multihead_timit/data' / (filename + '.jsonl')
        with path.open() as stream:
            timit[split] = {r['wav']: r['references']['ottc_arpabet']
                            for r in map(json.loads, stream)}
    for split, current in rows.items():
        missing = []
        frames = supervised = 0
        for row in current:
            intervals = timit[split][row['wav']] if split in timit else row['reference']
            targets = project(intervals, vocab, round(row['duration'] * 16000))
            ignored = int((targets == -100).sum())
            frames += len(targets)
            supervised += len(targets) - ignored
            if ignored:
                missing.append(dict(utt=row['utt'], ignored_frames=ignored))
            assert row['wav'] not in timelines
            timelines[row['wav']] = intervals
        coverage[split] = dict(utterances=len(current), frames=frames,
            supervised=supervised, ignored_frames=frames-supervised,
            utterances_with_ignored_frames=missing)
    processor.frame_ce_timelines = timelines
    processor.frame_ce_vocab = vocab
    processor.frame_ce_coverage = coverage
    data = {s: AudioDataset(rows[s], processor) for s in ('train', 'dev', 'test')}
    audit = dict(dataset_id='buckeye_full40_20261022', contracts=contracts,
        training_split='train', selection_split='dev',
        Buckeye_counts={s: len(rows[s]) for s in ('train', 'dev', 'test')},
        TIMIT_evaluation_only=True, inventory='ottc_legacy42', evaluation_uploads=False)
    return data, rows, vocab, processor, audit


build_model = parent.build_model
