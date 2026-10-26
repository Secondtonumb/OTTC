"""Independent full-training recipe; historical recipes remain unchanged."""
from reproduction.timit_ssl_native10_20261019.common import *
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.ottc_temporal_smoothing_20261021 import integration

HERE = Path(__file__).resolve().parent

def source_hashes():
    paths = [HERE / name for name in ('common.py', 'train.py', 'profile_recipe.py',
        'evaluate.py', 'interactive_gpu.sh', 'integration.py')]
    paths += [ROOT / 'ottc/ops_loss/temporal_smoothing.py',
        ROOT / 'ottc/models/sequence/wav2vec2_ottc_smoothing.py',
        ROOT / 'ottc/utils/temporal_smoothing_trainer.py']
    return dict(base.source_hashes(), **{str(p): sha(p) for p in paths})

def recipe(name, weight=.1, tau=4., epochs=40, batch_size=64):
    config = integration.recipe(name, weight, tau)
    config.update(epochs=epochs, global_batch=batch_size,
        checkpoint_policy='best three by development PER; latest full resumable state; no epoch averages',
        evaluation_uploads=False, source_sha256=source_hashes())
    config['name'] += f'__e{epochs}_b{batch_size}'
    return config

build_model = integration.build_model
