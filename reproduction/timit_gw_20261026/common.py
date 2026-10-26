"""Separate GW recipe; immutable historical baselines and resume contracts."""
from dataclasses import asdict
from pathlib import Path
import torch
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_seed_sweep_20261022.common import (
    ROOT, SSL, SSL_SHA, load_data, Collator, parameter_manifest, sha, write,
    phones_from_ids, normalize, scorer, atomic_weights, snapshot_sources,
)
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_gw import GWTransportMixin
from ottc.ops_loss.gw_transport import GWOptions

HERE = Path(__file__).resolve().parent


class GWScratchOTTC(GWTransportMixin, base.ScratchOTTC):
    pass


def source_hashes():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.pbs'))
    paths += [ROOT / 'ottc/ops_loss/gw_transport.py', ROOT / 'ottc/models/sequence/wav2vec2_ottc_gw.py',
              ROOT / 'reproduction/timit_seed_sweep_20261022/common.py',
              ROOT / 'reproduction/timit_seed_sweep_20261022/profile_recipe.py',
              ROOT / 'reproduction/timit_divergences_20261022/profile_recipe.py',
              ROOT / 'reproduction/timit_divergences_20261022/common.py']
    return dict(base.source_hashes(), **{str(p): sha(p) for p in paths})


def recipe(name):
    # Three independently seeded runs: mode 2, mode 3 and its zero-GW control.
    options = {'monotone_w0.1__seed43': GWOptions(mode='monotone'),
               'asot_w0.1__seed43': GWOptions(mode='asot'),
               'asot_w0__seed43': GWOptions(mode='asot', weight=0)}
    opt = options[name]
    config = base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    config.update(name=name, experiment_family=HERE.name, cost='ag', bins_metric='ag', seed=43,
                  baseline=str(ROOT / 'reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__jsd_upstream'),
                  historical_ag_seed43=str(ROOT / 'reproduction/timit_seed_sweep_20261022/models/ag__seed43'),
                  gw_options=asdict(opt), window_ms=20, evaluation_uploads=False,
                  pilot_epochs=5, temporal_smoothing_weight=0.,
                  objective='original AG on monotone plan + local occurrence GW' if opt.mode == 'monotone'
                  else 'unrolled ASOT-inspired AG + local occurrence GW + entropy + column KL + soft position prior',
                  target_structure='occurrence inequality; repeat phones retain separate target columns',
                  structure_normalization='0.5*N^2/valid_ordered_neighbor_pairs; raw transport mass',
                  asot_gradient='differentiate all finite solver iterations into logits AND learned frame mass',
                  asot_monotonicity='not guaranteed; soft positional prior only',
                  model_dtype='float32', average_dtype='float32',
                  source_sha256=source_hashes())
    return config


def build_model(config, vocab, steps_per_epoch):
    model = GWScratchOTTC(wvOTTCConfig(wav2vec2_path=str(SSL), vocab_size=42,
        bins_metric='ag', forward_type='regular', wav2vec2_output_dropout=.1,
        gw_options=config['gw_options']))
    return base.configure_model(model, config, vocab)


def retain_best(output, model, value, epoch, step, config):
    import json
    index = output / 'BEST_DEV.json'
    if index.exists() and value >= json.loads(index.read_text())['PER']:
        return
    path = output / 'best_dev.pt'
    atomic_weights(model, path, dict(epoch=epoch, step=step, recipe=config['name'],
        dev_PER=value, seed=config['seed'], model_config=model.config.to_dict(), source_sha256=config['source_sha256']))
    write(index, dict(PER=value, epoch=epoch, step=step, path=str(path), selection='development PER only'))
