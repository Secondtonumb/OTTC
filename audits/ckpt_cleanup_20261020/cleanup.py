"""Audit and prune old weights in four explicit POWSM runs; preserve resumes."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECIPES = ROOT / 'reproduction/powsm_bpe_timit_20261019/gradient_check_fix'
RUNS = ['ottc__jsd', 'ottc__ce', 'spottc_s3t2__jsd', 'spottc_s3t2__ce']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    info = path.stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size,
                allocated_bytes=info.st_blocks * 512, links=info.st_nlink, mtime_ns=info.st_mtime_ns)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def plan_run(run):
    metadata = json.loads((run / 'run_config.json').read_text())
    best = json.loads((run / 'BEST_DEV.json').read_text())
    completed = (run / 'BENCHMARK_COMPLETE.json').is_file()
    members = sorted((run / 'epoch_weights').glob('epoch-*.pt'))
    keep = set(members[-10:] + members[:1])
    best_epoch = run / 'epoch_weights' / f'epoch-{round(best["epoch"]):03d}.pt'
    if best_epoch.is_file():
        keep.add(best_epoch)
    for record in [run / 'RESULTS.json', *sorted((RECIPES / 'diagnostics').glob('*.json'))]:
        if record.exists():
            for value in strings(json.loads(record.read_text())):
                if value.endswith(('.pt', '.safetensors')):
                    referenced = Path(value).resolve()
                    if referenced.is_relative_to(run):
                        keep.add(referenced)
    candidates = []
    for path in members:
        if path not in keep:
            sidecar = path.with_suffix('.json')
            member = json.loads(sidecar.read_text())
            assert member['epoch'] == int(path.stem.split('-')[1])
            assert member.get('weights_sha256'), str(sidecar)
            candidates.append(dict(path=str(path.relative_to(ROOT)), reason='early epoch outside retained first/best/latest-ten/evaluation references',
                recorded_sha256=member['weights_sha256'], sha256_source=str(sidecar.relative_to(ROOT)), **signature(path)))
    if completed:
        latest = json.loads((run / 'LATEST_STATE.json').read_text())
        checkpoint = Path(latest['checkpoint']).resolve()
        assert latest['complete'] and checkpoint.parent == run and checkpoint.is_dir()
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert state['global_step'] == latest['step']
        assert all((checkpoint / name).is_file() for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth'])
        assert (checkpoint / 'model.safetensors').is_file()
        for older in sorted(run.glob('checkpoint-*')):
            if older == checkpoint:
                continue
            older_state = json.loads((older / 'trainer_state.json').read_text())
            assert older_state['global_step'] < latest['step']
            for name in ['model.safetensors', 'pytorch_model.bin', 'optimizer.pt']:
                path = older / name
                if path.is_file() and path.resolve() not in keep:
                    candidates.append(dict(path=str(path.relative_to(ROOT)), reason='redundant older full checkpoint of benchmark-completed run; latest full state retained',
                        recorded_sha256=None, sha256_source=None, **signature(path)))
    removed = {ROOT / item['path'] for item in candidates}
    assert not removed & keep
    retained = {}
    for path in run.rglob('*'):
        if path.is_file() and not path.is_symlink() and path not in removed and path.name != '.training.lock':
            record = signature(path)
            if record['size'] <= 5 * 1024 * 1024:
                record['sha256'] = sha256(path)
            retained[str(path.relative_to(ROOT))] = record
    assert (run / 'best_dev.pt') not in removed
    assert len([path for path in members if path not in removed]) >= min(10, len(members))
    return dict(run=str(run.relative_to(ROOT)), benchmark_complete=completed,
                configuration_sha256=sha256(run / 'run_config.json'),
                keep_epoch_weights=[str(path.relative_to(ROOT)) for path in sorted(keep) if path in members],
                candidates=candidates, retained=retained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert not (HERE / 'execution_result.json').exists(), 'This audited cleanup already executed'
    queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
    assert 'No unfinished job found' in queue, 'Jobs are active; do not run this cleanup'
    with ExitStack() as stack:
        plans, skipped = [], []
        for name in RUNS:
            run = RECIPES / 'models' / name
            lock = stack.enter_context((run / '.training.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                skipped.append(dict(run=str(run.relative_to(ROOT)), reason='training lock held'))
                continue
            plans.append(plan_run(run))
        candidates = [item for plan in plans for item in plan['candidates']]
        identities = set()
        allocated = 0
        for item in candidates:
            identity = (item['device'], item['inode'])
            if item['links'] == 1 and identity not in identities:
                allocated += item['allocated_bytes']
                identities.add(identity)
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), qstat=queue,
            policy='Only named POWSM runs; keep first, best epoch, latest ten, all evaluation references and best weights. All full checkpoints of unfinished runs retained. All BPE500/CTC/smoke/data/log/config/source files outside candidate list untouched.',
            planned_files=len(candidates), estimated_reclaim_bytes=allocated,
            estimate='Allocated blocks of uniquely linked candidate inodes; Lustre quota accounting can lag.',
            plans=plans, skipped=skipped)
        write_json(HERE / ('approved_manifest.json' if args.execute else 'plan.json'), manifest)
        print(json.dumps(dict(files=len(candidates), estimated_reclaim_GiB=allocated / 2**30,
            runs=[dict(run=plan['run'], completed=plan['benchmark_complete'], delete_files=len(plan['candidates']),
                       keep_epochs=plan['keep_epoch_weights']) for plan in plans], skipped=skipped), indent=2), flush=True)
        if not args.execute:
            return
        deleted = 0
        with (HERE / 'unlink_journal.jsonl').open('a') as journal:
            for item in candidates:
                path = ROOT / item['path']
                assert not path.is_symlink() and path.is_relative_to(RECIPES / 'models')
                assert signature(path) == {key: item[key] for key in signature(path)}, str(path)
                digest = item['recorded_sha256'] or sha256(path)
                journal.write(json.dumps(dict(event='before_unlink', path=item['path'], sha256=digest,
                    hash_kind='existing epoch metadata' if item['recorded_sha256'] else 'computed before deletion')) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                path.unlink()
                deleted += 1
                journal.write(json.dumps(dict(event='unlinked', path=item['path'])) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
        retained_count = 0
        for plan in plans:
            for relative, original in plan['retained'].items():
                path = ROOT / relative
                current = signature(path)
                assert current == {key: original[key] for key in current}, relative
                if 'sha256' in original:
                    assert sha256(path) == original['sha256'], relative
                retained_count += 1
        assert all(not (ROOT / item['path']).exists() for item in candidates)
        quota = subprocess.run(['lfs', 'quota', '-g', 'gm64', '/work'], capture_output=True, text=True, check=True).stdout
        write_json(HERE / 'execution_result.json', dict(deleted_files=deleted,
            verified_retained_files=retained_count, estimated_reclaim_bytes=allocated, quota_after=quota,
            evaluation_uploads=False, completed_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(deleted_files=deleted, verified_retained_files=retained_count, quota_after=quota), indent=2), flush=True)


if __name__ == '__main__':
    main()