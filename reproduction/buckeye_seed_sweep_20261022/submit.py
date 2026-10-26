"""Submit one twelve-node allocation, with one independent experiment per node."""
import argparse
import fcntl
import json
import os
import re
import subprocess

from reproduction.buckeye_seed_sweep_20261022.common import HERE, COSTS, SEEDS, recipe, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    with (HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = HERE / 'SUBMISSION.json'
        if receipt.exists():
            print(receipt.read_text(), flush=True)
            return
        configs = []
        for cost in COSTS:
            for seed in SEEDS:
                name = f'{cost}__seed{seed}'
                config = recipe(name)
                write(HERE / 'planned_configs' / f'{name}.json', config)
                configs.append(dict(rank=len(configs), recipe=name, config=str(HERE / 'planned_configs' / f'{name}.json')))
        write(HERE / 'EXPERIMENT_MATRIX.json', dict(experiments=configs, nodes=12, GPUs_per_node=1,
              independent=True, logical_batch_per_experiment=64, world_size_per_experiment=1))
        command = ['qsub', '-o', str(HERE / 'allocation.log'), str(HERE / 'run.pbs')]
        if not args.submit:
            print(json.dumps(dict(command=command, experiments=configs, submitted=False), indent=2))
            return
        result = subprocess.run(command, env=dict(os.environ, TMPDIR='/tmp'), text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'qsub failed: {result.stderr} {result.stdout}')
        matched = re.search(r'\b(\d+\.opbs)\b', result.stdout)
        if not matched:
            raise RuntimeError(f'Unrecognized PBS receipt; inspect before retry: {result.stdout}')
        state = dict(job_id=matched.group(1), command=command, queue='regular-g', nodes=12,
                     GPUs_per_node=1, walltime='06:00:00', status='submitted_not_completed', experiments=configs)
        write(receipt, state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
