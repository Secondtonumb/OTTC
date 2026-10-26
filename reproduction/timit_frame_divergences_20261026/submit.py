"""Submit one3GPU PBS job per loss; MPI dispatches three independent seeds."""
import json
import os
import re
import subprocess
import fcntl
from reproduction.timit_frame_divergences_20261026.common import HERE, COSTS, SEEDS, recipe, write, source_hashes


def main():
    assert json.loads((HERE / 'CPU_PASS.json').read_text())['source_sha256'] == source_hashes(), 'Run CPU validation first'
    assert json.loads((HERE / 'LAUNCHER_PASS.json').read_text())['source_sha256'] == source_hashes(), 'Run launcher validation first'
    cancelled = json.loads((HERE / 'REGROUP_CANCELLATION.json').read_text())
    assert cancelled['returncode'] == 0 and all(s == 'Q' for s in cancelled['states_before'].values())
    lock = (HERE / '.submit.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = HERE / 'SUBMISSIONS_GROUPED.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else dict(groups={}, supersedes='SUBMISSIONS.json')
    for cost in COSTS:
        names = [f'frame_{cost}__seed{seed}' for seed in SEEDS]
        for name in names:
            write(HERE / 'planned_configs' / f'{name}.json', recipe(name))
        if cost in state['groups']:
            print(cost, state['groups'][cost]['job_id'], 'already submitted', flush=True)
            continue
        log = HERE / f'group_{cost}.log'
        command = ['qsub', '-N', f'timit_{cost[:5]}3', '-v', f'FRAME_DIV_COST={cost}',
                   '-o', str(log), str(HERE / 'run.pbs')]
        result = subprocess.run(command, env=dict(os.environ, TMPDIR='/tmp'), text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'{cost}: {result.stderr} {result.stdout}')
        matched = re.search(r'\b(\d+\.opbs)\b', result.stdout)
        if not matched:
            raise RuntimeError(f'Unparsed scheduler receipt: {result.stdout}')
        state['groups'][cost] = dict(job_id=matched.group(1), cost=cost, seeds=list(SEEDS), recipes=names,
            GPUs=3, nodes=3, status='submitted', log=str(log), command=command)
        write(receipt, state)
        print(cost, matched.group(1), 'seeds42/43/44;3GPUs', flush=True)
    print(f"Submitted {len(state['groups'])}/3 grouped jobs, nine independent seeds.", flush=True)


if __name__ == '__main__':
    main()
