#!/bin/bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/../.."
source reproduction/libri960_continue5_20261017/environment.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OTTC_EVAL_UPLOAD=0 WANDB_RUN_GROUP=timit_bpe500_50ep_20261020
"$LIBRI_PYTHON" -u - "$script_dir" "$@" <<'PY'
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(sys.argv[1]).resolve()


def field(text, key):
    match = re.search(r'^\s*' + re.escape(key) + r'\s*=\s*(.*)$', text, re.MULTILINE)
    return match.group(1).strip() if match else ''


def clock(value):
    if not re.fullmatch(r'\d+:\d{2}:\d{2}', value):
        raise ValueError('Cannot determine PBS walltime: ' + repr(value))
    hours, minutes, seconds = map(int, value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def budget(limit, elapsed, requested):
    remaining = min(requested, clock(limit) - clock(elapsed) - 1200)
    if remaining < 180:
        raise RuntimeError('Insufficient remaining allocation after the 20-minute save/evaluation reserve')
    return remaining


def cancellation_plan(jobs, replace):
    for job in jobs:
        if job['state'] not in ('Q', 'H'):
            raise RuntimeError(f"Matching job {job['job']} is {job['state']}; refusing concurrent training")
        if job['dependents']:
            raise RuntimeError(f"Job {job['job']} has dependent jobs; inspect its queue chain manually")
    if jobs and not replace:
        raise RuntimeError('Matching queued jobs: ' + ', '.join(job['job'] for job in jobs) +
                           '. Use --replace-queued to cancel only these queued copies.')
    return [job['job'] for job in jobs]


def inspect_job(job):
    details = subprocess.run(['qstat', '-f', job], capture_output=True, text=True, check=True).stdout
    return dict(job=job, state=field(details, 'job_state'),
                dependents=bool(re.search(r'\bbefore(?:ok|any|notok)?:', details)))


def matching_jobs(recipe, own_job):
    manifest = ROOT / 'SUBMISSIONS.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    candidates = {record['job'].split('.')[0] for record in records if record['recipe'] == recipe}
    candidates.discard(own_job.split('.')[0])
    queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
    live = {line.split()[0].split('.')[0] for line in queue.splitlines()
            if line.split() and line.split()[0].split('.')[0].isdigit()}
    return [inspect_job(job) for job in sorted(candidates & live)]


def logged(command, log):
    with log.open('a') as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, close_fds=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()


def self_test():
    assert budget('02:00:00', '00:00:00', 6000) == 6000
    assert budget('02:00:00', '00:15:00', 6000) == 5100
    assert budget('04:00:00', '00:00:00', 6000) == 6000
    assert field('    job_state = R\n    Resource_List.walltime = 02:00:00', 'job_state') == 'R'
    queued = [dict(job='123', state='Q', dependents=False)]
    assert cancellation_plan(queued, True) == ['123']
    assert cancellation_plan([], False) == []
    failures = [lambda: budget('02:00:00', '01:50:00', 6000),
                lambda: cancellation_plan(queued, False),
                lambda: cancellation_plan([dict(job='123', state='R', dependents=False)], True),
                lambda: cancellation_plan([dict(job='123', state='Q', dependents=True)], True)]
    for operation in failures:
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unsafe launch condition was accepted')
    print('PASS: allocation budget, explicit queued replacement, running-job and dependency protection')


def main():
    parser = argparse.ArgumentParser(description='Interactive full TIMIT JSD: 50 epochs, physical batch64, resume supported.')
    parser.add_argument('--recipe', choices=['ottc_jsd', 'spottc_jsd'], default='ottc_jsd')
    parser.add_argument('--seconds', type=int, default=6000, help='Total invocation budget, capped by remaining PBS walltime')
    parser.add_argument('--replace-queued', action='store_true', help='Cancel matching Q/H batch jobs only; never running jobs')
    parser.add_argument('--self-test', action='store_true', help='CPU-only launcher checks; no queue changes')
    args = parser.parse_args(sys.argv[2:])
    if args.self_test:
        self_test()
        return 0
    if not 180 <= args.seconds <= 6000:
        parser.error('--seconds must be between 180 and 6000')
    job = os.environ.get('PBS_JOBID')
    if not job:
        raise RuntimeError('Run this script inside an allocated PBS GPU interactive terminal')
    if (ROOT / 'models' / args.recipe / 'BENCHMARK_COMPLETE.json').exists():
        print('This recipe already completed; no training started.')
        return 0
    details = subprocess.run(['qstat', '-f', job], capture_output=True, text=True, check=True).stdout
    if field(details, 'job_state') != 'R':
        raise RuntimeError('PBS allocation is not running')
    allowance = budget(field(details, 'Resource_List.walltime'), field(details, 'resources_used.walltime'), args.seconds)
    started = time.monotonic()
    jobs = ROOT / 'jobs'
    jobs.mkdir(exist_ok=True)
    with (jobs / (args.recipe + '.interactive.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cancel = cancellation_plan(matching_jobs(args.recipe, job), args.replace_queued)
        subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "No allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'], check=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log = jobs / f'interactive_{job}_{args.recipe}_{stamp}.log'
        print(f'Preflight: {args.recipe}, budget={allowance}s, log={log}', flush=True)
        result = logged([sys.executable, '-u', str(ROOT / 'train.py'), '--check'], log)
        if result:
            return result
        if allowance - (time.monotonic() - started) < 240:
            raise RuntimeError('Too little walltime remains after preflight; queued jobs were left untouched')
        for other in cancel:
            cancellation_plan([inspect_job(other)], True)
            subprocess.run(['qdel', other], check=True)
        remaining = int(allowance - (time.monotonic() - started))
        launcher = ROOT / 'run_jsd_interactive.sh'
        snapshot = jobs / f'interactive_{job}_{args.recipe}_{stamp}.sh'
        shutil.copy2(launcher, snapshot)
        record = dict(job=job, recipe=args.recipe, epochs=50, physical_batch=64,
            requested_seconds=args.seconds, training_budget_seconds=remaining, reserve_seconds=1200,
            replaced_queued_jobs=cancel, launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
            launcher_snapshot=str(snapshot), log=str(log), timestamp_utc=stamp, evaluation_uploads=False)
        metadata = log.with_suffix('.json')
        metadata.write_text(json.dumps(record, indent=2) + '\n')
        result = logged([sys.executable, '-u', str(ROOT / 'train.py'), '--recipe', args.recipe,
                         '--max-seconds', str(remaining)], log)
        record['exit_code'] = result
        metadata.write_text(json.dumps(record, indent=2) + '\n')
        if result == 75:
            print('Budget pause: checkpoint saved. Rerun this script to resume toward 50 epochs.')
        return result


try:
    sys.exit(main())
except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
    print(str(error), file=sys.stderr)
    sys.exit(2)
PY