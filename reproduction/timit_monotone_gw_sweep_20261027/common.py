"""Four existing costs x three matched seeds, with identical monotone GW."""
from dataclasses import asdict
from pathlib import Path
import json
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_seed_sweep_20261022.common import (
    ROOT, SSL, SSL_SHA, load_data, Collator, parameter_manifest, sha, write,
    phones_from_ids, normalize, scorer, atomic_weights, snapshot_sources,
    source_hashes as baseline_source_hashes,
)
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_monotone_gw import MonotoneGWTransportMixin
from ottc.ops_loss.gw_transport import GWOptions

HERE = Path(__file__).resolve().parent
COSTS = {'ce': 'crossentropy', 'ag': 'ag', 'jsd_standard': 'jsd_standard', 'jdiv': 'jdiv'}
SEEDS = (43, 44, 45)
REUSED_AG43 = ROOT/'reproduction/timit_gw_20261026/models/monotone_w0.1__seed43'


class GWScratchOTTC(MonotoneGWTransportMixin, base.ScratchOTTC):
    pass


def source_hashes():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.pbs'))
    paths += [ROOT/'ottc/ops_loss/gw_transport.py',
              ROOT/'ottc/models/sequence/wav2vec2_ottc_monotone_gw.py',
              ROOT/'reproduction/timit_divergences_20261022/profile_recipe.py']
    return dict(baseline_source_hashes(), **{str(p):sha(p) for p in paths})


def recipe(name):
    cost, seed = name.split('__seed'); seed = int(seed)
    if cost not in COSTS or seed not in SEEDS: raise ValueError(name)
    config = base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    baseline = ROOT/'reproduction/timit_seed_sweep_20261022/models'/name
    previous = json.loads((baseline/'run_config.json').read_text())
    fields = ('epochs','global_batch','learning_rate','warmup_steps','weight_decay','backbone',
              'frame_ms','head','forward_type','native_conv_stride','target_mass',
              'freeze_cnn_first_six','train_cnn_last_layer','train_transformer','train_all_task_heads')
    assert all(config[key] == previous[key] for key in fields)
    config.update(name=name, experiment_family=HERE.name, cost=cost, bins_metric=COSTS[cost], seed=seed,
        baseline=str(baseline), gw_options=asdict(GWOptions(mode='monotone')),
        window_ms=20, evaluation_uploads=False, temporal_smoothing_weight=0., pilot_epochs=5,
        objective=f'unchanged upstream {COSTS[cost]} on original monotone plan + 0.1 local occurrence GW',
        target_structure='occurrence inequality; repeated phones retain distinct columns',
        structure_normalization='0.5*N^2/valid_ordered_neighbor_pairs; raw transport mass',
        weight_selection='fixed 0.1 for every cost, no test tuning; differing base-cost scales acknowledged',
        model_dtype='float32',average_dtype='float32',source_sha256=source_hashes())
    return config


def build_model(config,vocab,steps_per_epoch):
    model=GWScratchOTTC(wvOTTCConfig(wav2vec2_path=str(SSL),vocab_size=42,
        bins_metric=config['bins_metric'],forward_type='regular',wav2vec2_output_dropout=.1,
        gw_options=config['gw_options']))
    return base.configure_model(model,config,vocab)


def retain_best(output,model,value,epoch,step,config):
    index=output/'BEST_DEV.json'
    if index.exists() and value>=json.loads(index.read_text())['PER']:return
    path=output/'best_dev.pt'
    atomic_weights(model,path,dict(epoch=epoch,step=step,recipe=config['name'],dev_PER=value,
        seed=config['seed'],model_config=model.config.to_dict(),source_sha256=config['source_sha256']))
    write(index,dict(PER=value,epoch=epoch,step=step,path=str(path),selection='development PER only'))
