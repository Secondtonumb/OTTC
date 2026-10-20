"""Submit two ordered experiment chains and reject source/data drift on resume."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess

from train import HERE, source_hashes
from reproduction.powsm_bpe_timit_20261019.prepare import sha256, write_json


def identity():
    return dict(source_sha256=source_hashes(),
        metadata_sha256=sha256(HERE / 'data/metadata.json'),
        matrix=[dict(head=head, cost=cost) for head in ['ottc', 'spottc_s3t2'] for cost in ['jsd', 'ce']],
        order_per_cost=['ottc', 'spottc_s3t2'], evaluation_uploads=False)


def submit(cost, parent_job=None):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log = HERE / 'jobs' / f'{cost}_{stamp}.log'
    command = ['/opt/pbs/bin/qsub', '-N', f'powsm_{cost}', '-v', f'POWSM_COST={cost}',
               '-o', str(log), str(HERE / 'run.pbs')]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    record = dict(cost=cost, job=result.stdout.strip(), parent_job=parent_job,
        submitted_utc=stamp, log=str(log), command=command, stderr=result.stderr)
    write_json(HERE / 'jobs' / f'{cost}_{stamp}.json', record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--submit', action='store_true')
    action.add_argument('--verify', action='store_true')
    action.add_argument('--continue-cost', choices=['jsd', 'ce'])
    args = parser.parse_args()
    jobs = HERE / 'jobs'
    jobs.mkdir(exist_ok=True)
    lock = (jobs / '.dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = HERE / 'DISPATCH.json'
    current = identity()
    if args.submit:
        assert not manifest.exists(), 'Already dispatched; inspect saved jobs before any manual resubmission'
        write_json(manifest, current)
        for cost in ['jsd', 'ce']:
            submit(cost)
    else:
        assert json.loads(manifest.read_text()) == current, 'Source/data contract changed; do not resume'
        if args.continue_cost:
            parent = os.environ['PBS_JOBID']
            assert os.environ['POWSM_COST'] == args.continue_cost
            existing = [json.loads(path.read_text()) for path in jobs.glob('*.json')]
            assert not any(record['parent_job'] == parent for record in existing), 'Continuation already submitted'
            submit(args.continue_cost, parent)
        else:
            print('PASS: dispatch source and data contract unchanged')


if __name__ == '__main__':
    main()