"""Opt-in model factory for the existing fresh-SSL/native-10ms OTTC recipe.

Uses the original recipe's data, initialization, CNN trainability and selected
OTTC cost. No historical trainer, recipe, checkpoint or source hash is edited.
"""
import hashlib
import math
from pathlib import Path

from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_smoothing import TemporalSmoothingMixin
from reproduction.timit_ssl_native10_20261019 import common as base

HERE = Path(__file__).resolve().parent


class SmoothedScratchOTTC(TemporalSmoothingMixin, base.ScratchOTTC):
    """Preserve the original native-stride/device adaptation with smoothing."""


def recipe(base_name, smooth_weight, tau=4.0):
    if not math.isfinite(smooth_weight) or smooth_weight < 0:
        raise ValueError('smooth_weight must be finite and nonnegative')
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    config = base.recipe(base_name)
    config.update(name=f'{base_name}__temporal_w{smooth_weight:g}_tau{tau:g}',
        experiment_family=HERE.name, base_recipe=base_name,
        temporal_smoothing_weight=float(smooth_weight), temporal_smoothing_tau=float(tau),
        temporal_smoothing_reduction='utterance_mean_then_batch_sum',
        objective='original selected OTTC cost + temporal smoothing; per-utterance means summed over batch')
    paths = [Path(__file__), base.ROOT / 'ottc/ops_loss/temporal_smoothing.py',
        base.ROOT / 'ottc/models/sequence/wav2vec2_ottc_smoothing.py',
        base.ROOT / 'ottc/utils/temporal_smoothing_trainer.py']
    config['source_sha256'] = dict(config['source_sha256'],
        **{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return config


def build_model(config, vocab, steps_per_epoch):
    model = SmoothedScratchOTTC(wvOTTCConfig(wav2vec2_path=str(base.SSL), vocab_size=42,
        bins_metric=config['bins_metric'], forward_type='regular', wav2vec2_output_dropout=.1,
        temporal_smoothing_weight=config['temporal_smoothing_weight'],
        temporal_smoothing_tau=config['temporal_smoothing_tau']))
    return base.configure_model(model, config, vocab)
