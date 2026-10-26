"""Matched 10ms TIMIT AG baseline plus the frozen IF-MDD CROTTC recipe."""
from pathlib import Path
import json

import torch

from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, SSL, SSL_SHA, load_data, Collator, sha, write, phones_from_ids, normalize, scorer,
)
from reproduction.crottc_timit_20ms_20261019 import common as historical
from reproduction.crottc_timit_20ms_20261019.common import CROTTC, SpeechViews, pair_chunks
from reproduction.timit_seed_sweep_20261022.common import snapshot_sources
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig

HERE = Path(__file__).resolve().parent
SEEDS = (42,43,44,45)


def source_hashes():
    paths = [HERE/name for name in ('common.py','train.py','profile_recipe.py','evaluate.py','validate.py','report.py','run.pbs')]
    paths += [ROOT/'reproduction/timit_seed_sweep_20261022/pfer.py',
              ROOT/'reproduction/timit_seed_sweep_20261022/report.py',
              ROOT/'reproduction/timit_seed_sweep_20261022/common.py',
              ROOT/'reproduction/timit_divergences_20261022/common.py',
              ROOT.parent/'fa-bench/analysis/panphon_pfer/distance.py',
              ROOT.parent/'fa-bench/analysis/panphon_pfer/feature_distance.so',
              ROOT.parent/'seamless_communication/outputs/panphon_libri100/panphon_table.json']
    return dict(base.source_hashes(), **historical.source_hashes(), **{str(p):sha(p) for p in paths})


def matrix():
    return [f'crottc_ag__seed{seed}' for seed in SEEDS]


def baseline_path(seed):
    if seed == 42:
        return ROOT/'reproduction/timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__jsd_upstream'
    return ROOT/'reproduction/timit_seed_sweep_20261022/models'/f'ag__seed{seed}'


def recipe(name):
    assert name in matrix(),name
    seed = int(name.split('__seed')[1])
    config = base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    old = historical.recipe('wav2vec2__timit__20ms__mono__crottc_jsd_upstream')
    config.update(name=name, experiment_family=HERE.name, cost='crottc_ag', bins_metric='ag', seed=seed,
        cr_loss_weight=old['cr_loss_weight'], cr_reduction=old['cr_reduction'], augmentation=old['augmentation'],
        objective=old['objective'], baseline=str(baseline_path(seed)),
        comparison_contract='same seed/data/SSL/native10ms/trainability/optimizer/best-dev FP32 scorer as four-seed AG table; adds shared-speed dual augmentation plus legacy CR',
        evaluation_uploads=False, parameter_dtype='float32', averaging_dtype='float32', saved_weights_dtype='float32',
        checkpoint_policy='independent FP32 weights every epoch, retain all40; best-dev and full latest resume; final5/final10 FP32 averages separate from single-best-dev headline',
        source_sha256=source_hashes())
    previous = json.loads((baseline_path(seed)/'run_config.json').read_text())
    for field in ('epochs','global_batch','learning_rate','warmup_steps','weight_decay','seed','backbone','frame_ms','head',
                  'forward_type','native_conv_stride','target_mass','freeze_cnn_first_six','train_cnn_last_layer',
                  'train_transformer','train_all_task_heads'):
        assert config[field] == previous[field], (field,config[field],previous[field])
    return config


def build_model(config,vocab,steps_per_epoch):
    model = CROTTC(wvOTTCConfig(wav2vec2_path=str(SSL),vocab_size=len(vocab),bins_metric='ag',
                              forward_type='regular',wav2vec2_output_dropout=.1))
    model.config.cr_loss_weight=config['cr_loss_weight']
    model,topo=base.configure_model(model,config,vocab)
    assert topo is None and model.divergence=='ag'
    assert all(p.dtype==torch.float32 for p in model.parameters())
    return model,topo


def parameter_manifest(model):
    return base.parameter_manifest(model)


def atomic_weights(model,path,metadata):
    assert all(p.dtype==torch.float32 for p in model.parameters())
    base.atomic_weights(model,path,metadata)
