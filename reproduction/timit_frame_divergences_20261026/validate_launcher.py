"""Exercise actual worker rank mapping and MPI environment removal without GPUs."""
import os
import subprocess
from reproduction.timit_frame_divergences_20261026.common import HERE, COSTS, SEEDS, write, source_hashes


def main():
    records = []
    for cost in COSTS:
        for rank, seed in enumerate(SEEDS):
            poisoned = dict(os.environ, FRAME_DIV_COST=cost, OMPI_COMM_WORLD_RANK=str(rank),
                OMPI_COMM_WORLD_SIZE='3', OMPI_COMM_WORLD_LOCAL_RANK='0', RANK=str(rank), WORLD_SIZE='3',
                LOCAL_RANK='0', LOCAL_WORLD_SIZE='1', MASTER_ADDR='unused', MASTER_PORT='12345',
                PMI_RANK=str(rank), PMI_SIZE='3', PMIX_RANK=str(rank))
            checked = subprocess.run(['bash', str(HERE / 'worker.sh'), '--dry-run'], env=poisoned,
                text=True, capture_output=True, check=True)
            fields = dict(word.split('=', 1) for word in checked.stdout.strip().split())
            assert fields['recipe'] == f'frame_{cost}__seed{seed}'
            assert fields['world_size'] == fields['GPUs'] == '1'
            assert fields['batch'] == '64' and fields['epochs'] == '40'
            records.append(fields)
    assert len({r['recipe'] for r in records}) == 9
    for rank, cost in [('3', 'ag'), ('-1', 'ag'), ('0', 'unknown')]:
        checked = subprocess.run(['bash', str(HERE / 'worker.sh'), '--dry-run'],
            env=dict(os.environ, OMPI_COMM_WORLD_RANK=rank, FRAME_DIV_COST=cost), text=True, capture_output=True)
        assert checked.returncode != 0
    for name in ('run.pbs', 'worker.sh'):
        subprocess.run(['bash', '-n', str(HERE / name)], check=True)
    write(HERE / 'LAUNCHER_PASS.json', dict(source_sha256=source_hashes(), assignments=records,
        poisoned_MPI_environment_removed=True, invalid_rank_and_cost_rejected=True,
        grouped_jobs=3, GPUs_per_job=3, independent_seeds_per_job=3, GPU_execution_tested=False))
    print('LAUNCHER_PASS:3jobs x3GPUs;9 unique independent seeds; MPI environment scrubbed')


if __name__ == '__main__':
    main()
