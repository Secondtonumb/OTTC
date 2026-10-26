"""Matched frame JSD/J-DIV/AG; reuse the active CE data contract unchanged."""
from pathlib import Path
import torch
from reproduction.timit_frame_ce_20261026 import common as ce
from reproduction.timit_frame_ce_20261026.common import (
    ROOT, SSL, SSL_SHA, sha, write, phones_from_ids, normalize, scorer, parameter_manifest,
    load_data, Collator, frame_count, project, atomic_weights, snapshot_sources, retain_best,
)
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_frame_divergences_20261026.model import FrameDivergence

HERE = Path(__file__).resolve().parent
COSTS = ('jsd_standard', 'jdiv', 'ag')
SEEDS = (42, 43, 44)


def source_hashes():
    files = list(HERE.glob('*.py')) + [HERE / 'run.pbs', HERE / 'worker.sh']
    return dict(ce.source_hashes(), **{str(p): sha(p) for p in sorted(files)})


def recipe(name):
    prefix, seed_text = name.split('__seed')
    assert prefix.startswith('frame_')
    cost = prefix.removeprefix('frame_')
    seed = int(seed_text)
    assert cost in COSTS and seed in SEEDS
    config = ce.recipe(f'frame_ce__seed{seed}')
    definitions = dict(
        jsd_standard='standard JS: 0.5 KL(p||m)+0.5 KL(q||m), m=(p+q)/2; exact one-hot q',
        jdiv='Jeffreys J: KL(p||q_eps)+KL(q_eps||p), no half; q_eps=normalize(clamp(q,min=1e-12))',
        ag='exact upstream legacy jsd AG: 0.5 KL(m||p)+0.5 KL(m||q_perturbed); original unnormalized epsilon perturbation')
    config.update(name=name, cost=cost, bins_metric=cost, frame_divergence=cost,
        experiment_family=HERE.name, objective=definitions[cost],
        loss_reduction='sum classes; mean supervised frames per utterance; mean actual logical-batch utterances',
        divergence_arithmetic='FP32; natural logarithms; original repository helper called directly',
        one_hot_regularization=('none' if cost == 'jsd_standard' else
            'normalize(clamp_min(q,1e-12))' if cost == 'jdiv' else 'q + (1-q)*1e-12 - q*1e-12; no renormalization'),
        parallelism='one3GPU/3node PBS job per objective; MPI ranks0/1/2 dispatch independent seeds42/43/44; world_size=1 per seed; three jobs total',
        source_sha256=source_hashes())
    return config


def build_model(config, vocab, steps_per_epoch):
    model_config = base.wvOTTCConfig(wav2vec2_path=str(SSL), vocab_size=42,
        bins_metric=config['bins_metric'], forward_type='regular', wav2vec2_output_dropout=.1,
        frame_divergence=config['frame_divergence'])
    model = FrameDivergence(model_config)
    model, topo = base.configure_model(model, config, vocab)
    model.lm_weight.requires_grad_(False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model, topo
