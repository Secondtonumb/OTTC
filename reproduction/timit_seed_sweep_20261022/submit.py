"""Idempotent submission of the explicitly requested thirteen single-GPU runs."""
import json
import os
from pathlib import Path
import re
import subprocess

from reproduction.timit_seed_sweep_20261022.common import HERE, recipe, write


def main():
    receipt = HERE / 'SUBMISSIONS.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else dict(jobs={})
    matrix = [('jdiv', seed) for seed in (42, 43, 44, 45)]
    matrix += [(cost, seed) for cost in ('ce', 'jsd_standard', 'ag') for seed in (43, 44, 45)]
    environment = dict(os.environ, TMPDIR='/tmp')
    for cost, seed in matrix:
        name = f'{cost}__seed{seed}'
        config = recipe(name)
        write(HERE / 'planned_configs' / f'{name}.json', config)
        if name in state['jobs']:
            print(name, state['jobs'][name]['job_id'], 'already submitted', flush=True)
            continue
        log = HERE / f'{cost}_seed{seed}.log'
        command = ['qsub', '-N', f'timit_{cost[:6]}_s{seed}', '-v', f'SEED_RECIPE={name}',
                   '-o', str(log), str(HERE / 'run.pbs')]
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'{name}: qsub failed: {result.stderr} {result.stdout}')
        matched = re.search(r'\b(\d+\.opbs)\b', result.stdout)
        if not matched:
            raise RuntimeError(f'Cannot parse scheduler receipt for {name}: {result.stdout}')
        entry = dict(job_id=matched.group(1), cost=cost, seed=seed, status='submitted',
                     log=str(log), command=command, output_dir=str(HERE / 'models' / name))
        state['jobs'][name] = entry
        write(receipt, state)
        print(name, entry['job_id'], flush=True)
    print(f"Submitted {len(state['jobs'])}/13 requested experiments.", flush=True)


if __name__ == '__main__':
    main()
