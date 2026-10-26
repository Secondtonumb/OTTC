"""TIMIT frame-supervised companion to the completed OTTC seed OTTC sweep."""
import json
from pathlib import Path
import numpy as np
import torch
from reproduction.timit_seed_sweep_20261022 import common as sweep
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, sha, write, phones_from_ids, normalize, scorer, parameter_manifest,
)
from reproduction.timit_frame_ce_20261026.model import FrameCE

HERE = Path(__file__).resolve().parent
KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 1)


def source_hashes():
    files = list(HERE.glob('*.py')) + [HERE / 'run.pbs', ROOT / 'reproduction/segmentation_metrics.py',
        ROOT / 'reproduction/wandb_policy.py', ROOT / 'reproduction/libri960_continue5_20261017/environment.sh']
    return dict(sweep.source_hashes(), **{str(p): sha(p) for p in sorted(files)})


def recipe(name):
    prefix, seed = name.split('__seed')
    assert prefix == 'frame_ce' and int(seed) in (42, 43, 44)
    config = sweep.recipe(f'ce__seed{seed}')
    config.update(name=name, seed=int(seed), cost='frame_ce', experiment_family=HERE.name,
        objective='gold TIMIT PHN hard frame cross entropy; mean valid frames per utterance, then mean utterances',
        frame_supervision='references.ottc_arpabet including sil/q/dx; native CNN center in [start,end); uncovered centers ignored, never filled',
        frame_center_samples='199.5 + 160 * frame_index at 16kHz (400-sample receptive field)',
        target_mass=None, mass_head_training='frozen and unused; no OT/CTC/auxiliary loss',
        train_all_task_heads=False, train_phone_head=True, evaluation_uploads=False,
        checkpoint_policy='atomic independent weights every completed epoch; retain all40; best-dev symlink; latest full resume each epoch/interruption',
        comparison_contract='primary: same FP32 single best-dev PER as sweep; final5/final10 separately; dev-only selection',
        source_sha256=source_hashes())
    return config


def frame_count(samples):
    for kernel, stride in zip(KERNELS, STRIDES):
        samples = (samples - kernel) // stride + 1
    return samples


def project(intervals, vocab, samples):
    n = frame_count(samples)
    assert n > 0
    centers = (199.5 + np.arange(n) * 160) / 16000
    target = np.full(n, -100, dtype=np.int64)
    previous = 0.
    for phone, start, end in intervals:
        if phone not in vocab or phone in ('<pad>', '<unk>'):
            raise ValueError(f'Unknown or non-phone target: {phone}')
        assert start >= previous - 1e-7 and end > start, (phone, start, previous, end)
        inside = (centers >= start) & (centers < end)
        assert np.all(target[inside] == -100)
        target[inside] = vocab[phone]
        previous = end
    assert previous <= samples / 16000 + 1e-6
    assert np.any(target >= 0), 'No supervised frame in utterance'
    return target


def load_data(config):
    data, rows, vocab, processor, audit = base.load_data(config)
    timelines = {}
    coverage = {}
    manifest_root = ROOT / 'reproduction/multihead_timit/data'
    for split, filename in (('train', 'train-train'), ('dev', 'train-dev'), ('test', 'test')):
        original = [json.loads(line) for line in (manifest_root / f'{filename}.jsonl').open()]
        missing = []
        total_frames = 0
        total_supervised = 0
        for row, source in zip(rows[split], original):
            intervals = source['references']['ottc_arpabet']
            assert [vocab[p] for p, _, _ in intervals] == row['training_target_ids']
            samples = round(row['duration'] * 16000)
            targets = project(intervals, vocab, samples)
            count = int((targets == -100).sum())
            total_frames += len(targets)
            total_supervised += len(targets) - count
            if count:
                missing.append(dict(utt=row['utt'], ignored_frames=count, first_label_start=intervals[0][1], last_label_end=intervals[-1][2]))
            timelines[row['wav']] = intervals
        coverage[split] = dict(frames=total_frames, supervised=total_supervised,
                               ignored_frames=total_frames-total_supervised, utterances_with_ignored_frames=missing)
    # Existing evaluator constructs Collator(processor) for each utterance.
    processor.frame_ce_timelines = timelines
    processor.frame_ce_vocab = vocab
    processor.frame_ce_coverage = coverage
    return data, rows, vocab, processor, audit


class Collator(base.Collator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        samples = batch['attention_mask'].sum(-1).tolist()
        frames = torch.full((len(examples), frame_count(batch['input_values'].shape[1])), -100, dtype=torch.long)
        for index, (row, length) in enumerate(zip(examples, samples)):
            target = project(self.processor.frame_ce_timelines[row['wav']], self.processor.frame_ce_vocab, length)
            frames[index, :len(target)] = torch.from_numpy(target)
        batch['frame_targets'] = frames
        return batch


def build_model(config, vocab, steps_per_epoch):
    model = FrameCE(base.wvOTTCConfig(wav2vec2_path=str(SSL), vocab_size=42,
        bins_metric='crossentropy', forward_type='regular', wav2vec2_output_dropout=.1))
    model, topo = base.configure_model(model, config, vocab)
    model.lm_weight.requires_grad_(False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model, topo


def atomic_weights(model, path, metadata):
    assert all(p.dtype == torch.float32 for p in model.parameters())
    base.atomic_weights(model, path, metadata)


snapshot_sources = sweep.snapshot_sources


def retain_best(output, model, value, epoch, step, config):
    previous_path = output / 'BEST_DEV.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous is not None and value >= previous['PER']:
        return
    path = output / 'epoch_weights' / f'epoch-{int(round(epoch)):03d}.pt'
    assert path.exists(), 'Epoch weights must commit before dev selection'
    link = output / '.best_dev.tmp'
    link.unlink(missing_ok=True)
    link.symlink_to(path.relative_to(output))
    link.replace(output / 'best_dev.pt')
    write(previous_path, dict(PER=value, epoch=epoch, step=step, path=str(path),
                             selection='minimum epoch development PER; test unused'))
