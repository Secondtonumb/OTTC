import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent


def test_twelve_independent_worker_assignments():
    assignments = []
    for rank in range(12):
        result = subprocess.run(['bash', str(HERE / 'worker.sh'), '--dry-run'],
                                env=dict(os.environ, OMPI_COMM_WORLD_RANK=str(rank)),
                                text=True, capture_output=True, check=True)
        fields = dict(field.split('=', 1) for field in result.stdout.strip().split())
        assert fields['batch'] == '64'
        assert fields['GPUs'] == fields['world_size'] == '1'
        assert fields['epochs'] == '40'
        assignments.append(fields['recipe'])
    assert set(assignments) == {f'{cost}__seed{seed}' for cost in ('ce', 'jsd_standard', 'ag', 'jdiv') for seed in (42, 43, 44)}


def test_invalid_worker_rank_rejected():
    result = subprocess.run(['bash', str(HERE / 'worker.sh'), '--dry-run'],
                            env=dict(os.environ, OMPI_COMM_WORLD_RANK='12'), capture_output=True)
    assert result.returncode != 0


def test_existing_data_contract_matches_cached_recipe():
    from reproduction.buckeye_seed_sweep_20261022.common import recipe
    config = recipe('ag__seed43')
    assert config['expected_data']['counts'] == dict(train=10927, dev=1524, test=1840)
    assert config['global_batch'] == 64
    assert config['epochs'] == 40
    assert config['corpus'] == 'buckeye'
    assert config['bins_metric'] == 'ag'
    assert config['forward_type'] == 'regular'
    assert config['native_conv_stride'] == [5, 2, 2, 2, 2, 2, 1]
    assert config['freeze_cnn_first_six'] and config['train_cnn_last_layer']
    assert '19' in config['dataset_scope']
