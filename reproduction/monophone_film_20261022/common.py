"""Fresh SSL-initialized TIMIT monophone CTC + FiLM + OTTC-AG recipe."""
import json
from pathlib import Path

import torch

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, load_data, Collator, sha, write, atomic_weights,
    phones_from_ids, normalize, scorer,
)
from reproduction.timit_divergences_20261022.common import snapshot_sources
from ottc.models.sequence.wav2vec2_ottc_film import wvForOTTCFiLM, wvOTTCFiLMConfig

HERE = Path(__file__).resolve().parent
NAME = 'timit__mono_film__ctc_1__ottc_ag_1__seed42'


def source_hashes():
    paths = [HERE / name for name in ('common.py', 'train.py', 'profile_recipe.py', 'evaluate.py', 'run.pbs')]
    paths += [ROOT / 'ottc/models/sequence/wav2vec2_ottc_film.py',
              ROOT / 'reproduction/timit_divergences_20261022/common.py',
              ROOT / 'reproduction/libri960_continue5_20261017/environment.sh']
    return dict(base.source_hashes(), **{str(p): sha(p) for p in paths})


def recipe(name):
    assert name == NAME, name
    config = base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    config.update(name=name, experiment_family=HERE.name, cost='ag', bins_metric='ag',
        objective='CTC batch sum + native upstream arithmetic-geometric OTTC batch sum; backward divides by actual logical batch size',
        ctc_loss_weight=1., ottc_loss_weight=1., film_detach_condition=True,
        film_condition='audio-only monophone CTC posterior, detached as in original ContextFiLM',
        initialization='original facebook/wav2vec2-large-lv60 SSL; fresh CTC/OTTC/mass heads; identity-initialized FiLM',
        evaluation_uploads=False, parameter_dtype='float32', checkpoint_dtype='float32', averaging_dtype='float32',
        checkpoint_policy='all40 independent epoch weights + full latest Trainer state + best-dev; final5/final10 FP32 averages',
        source_sha256=source_hashes())
    return config


def build_model(config, vocab, steps_per_epoch):
    model = wvForOTTCFiLM(wvOTTCFiLMConfig(wav2vec2_path=str(SSL), vocab_size=len(vocab),
        bins_metric='ag', wav2vec2_output_dropout=.1, blank_id=vocab['<pad>'],
        ctc_loss_weight=config['ctc_loss_weight'], ottc_loss_weight=config['ottc_loss_weight'],
        film_detach_condition=config['film_detach_condition']))
    model, topo = base.configure_model(model, config, vocab)
    assert topo is None
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.requires_grad for module in (model.ctc_head, model.context_film) for p in module.parameters())
    return model, topo


def parameter_manifest(model):
    result = base.parameter_manifest(model)
    for name in ('ctc_head', 'context_film'):
        result['heads'][name] = {n:dict(shape=list(p.shape), trainable=p.requires_grad, dtype=str(p.dtype))
                                  for n,p in getattr(model,name).named_parameters()}
    return result
