"""Conservative cross-recipe cleanup of completed recent OTTC experiments."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from audits.ckpt_cleanup_20261020 import cleanup as audit

REPRODUCTION = ROOT / 'reproduction'
SKIP_DIRS = {'vendor', 'runtime', 'conda_pkgs', 'pip_cache', 'cache', 'data', 'datasets',
             'source_snapshot', 'source_snapshots', 'wandb', '__pycache__', 'epoch_weights',
             'smoke', 'profiles', 'validation', 'provenance', 'snapshots', 'import_source'}
EXCLUDED = {'timit_bpe500_20261020', 'powsm_bpe_timit_20261019', 'librispeech_bpe500_20261020'}
DATE = re.compile(r'202610(?:1[3-9]|20)')
WEIGHT = re.compile(r'(?:\.pt|\.bin|\.safetensors)$|(?:^|/)checkpoint-\d+$')


def inventory():
    runs, references, errors = [], {}, []
    for directory, children, filenames in os.walk(REPRODUCTION, followlinks=False):
        current = Path(directory)
        if 'run_config.json' in filenames and DATE.search(str(current.relative_to(REPRODUCTION))):
            if not EXCLUDED.intersection(current.parts):
                runs.append(current)
        children[:] = [name for name in children if name not in SKIP_DIRS and
                       not name.startswith(('checkpoint-', '.')) and not (current / name).is_symlink()]
        for filename in filenames:
            path = current / filename
            if path.suffix != '.json' or path.is_symlink() or path.stat().st_size > 16 * 1024**2:
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError, OSError) as error:
                errors.append(dict(path=str(path.relative_to(ROOT)), error=str(error)))
                continue
            for value in audit.strings(record):
                if not WEIGHT.search(value):
                    continue
                if value.startswith('/'):
                    target = Path(value).resolve()
                elif value.startswith('reproduction/'):
                    target = (ROOT / value).resolve()
                else:
                    target = (current / value).resolve()
                if target.is_relative_to(REPRODUCTION):
                    references.setdefault(target, set()).add(str(path.relative_to(ROOT)))
    return sorted(set(runs)), references, errors


def referenced(path, references):
    return any(target in references for target in [path, *path.parents])


def validate_weights(path):
    if path.suffix == '.safetensors':
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            assert 0 < length < 16 * 1024**2, 'invalid safetensors header length'
            header = json.loads(stream.read(length))
        offsets = [entry['data_offsets'] for name, entry in header.items() if name != '__metadata__']
        assert offsets and all(0 <= start <= end for start, end in offsets)
        assert max(end for _, end in offsets) + 8 + length == path.stat().st_size, 'truncated safetensors payload'
    else:
        with zipfile.ZipFile(path) as archive:
            assert archive.namelist(), 'empty torch checkpoint archive'


def plan_run(run, references, errors):
    assert (run / 'TRAIN_COMPLETE.json').is_file(), 'no completed-training marker'
    assert not any((ROOT / item['path']).is_relative_to(run) for item in errors), 'unreadable run JSON'
    config = json.loads((run / 'run_config.json').read_text())
    complete = json.loads((run / 'TRAIN_COMPLETE.json').read_text())
    best = json.loads((run / 'BEST_DEV.json').read_text())
    checkpoints = sorted(run.glob('checkpoint-*'), key=lambda path: int(path.name.split('-')[1]))
    assert checkpoints, 'no Trainer checkpoints; format not handled'
    latest_record = json.loads((run / 'LATEST_STATE.json').read_text()) if (run / 'LATEST_STATE.json').exists() else complete
    latest = Path(latest_record['checkpoint']).resolve()
    assert latest.parent == run and latest in checkpoints, 'latest checkpoint missing or external'
    state = json.loads((latest / 'trainer_state.json').read_text())
    expected_step = complete.get('step', complete.get('global_step'))
    assert expected_step is not None and state['global_step'] == expected_step, 'completion/latest step mismatch'
    assert state['global_step'] == int(latest.name.split('-')[1]), 'checkpoint step mismatch'
    model_paths = [latest / name for name in ['model.safetensors', 'pytorch_model.bin'] if (latest / name).is_file()]
    assert len(model_paths) == 1, 'ambiguous latest model payload'
    validate_weights(model_paths[0])
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']:
        validate_weights(latest / name)
    epochs = sorted((run / 'epoch_weights').glob('epoch-*.pt'), key=lambda path: int(path.stem.split('-')[1]))
    keep = set(epochs[:1] + epochs[-10:])
    if 'epoch' in best:
        keep.add(run / 'epoch_weights' / f'epoch-{round(best["epoch"]):03d}.pt')
    best_checkpoint = state.get('best_model_checkpoint')
    local_references = dict(references)
    if best_checkpoint:
        path = Path(best_checkpoint)
        path = path.resolve() if path.is_absolute() else (run / path).resolve()
        local_references.setdefault(path, set()).add(str(latest / 'trainer_state.json'))
    candidates = []
    for path in epochs:
        if path in keep or referenced(path, local_references):
            keep.add(path)
            continue
        digest = None
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            member = json.loads(sidecar.read_text())
            assert int(member['epoch']) == int(path.stem.split('-')[1])
            digest = member.get('weights_sha256')
        candidates.append(dict(path=str(path.relative_to(ROOT)), reason='unreferenced early epoch; first/best/newest-ten retained',
                               recorded_sha256=digest, **audit.signature(path)))
    protected_checkpoints = {latest}
    if len(checkpoints) > 2:
        protected_checkpoints.update(checkpoints[-10:])
    for checkpoint in checkpoints:
        if checkpoint in protected_checkpoints or referenced(checkpoint, local_references):
            continue
        older = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert older['global_step'] < expected_step
        for name in ['model.safetensors', 'pytorch_model.bin', 'optimizer.pt']:
            path = checkpoint / name
            if path.is_file() and not referenced(path, local_references):
                candidates.append(dict(path=str(path.relative_to(ROOT)), reason='older redundant Trainer payload; completed latest/full state retained',
                                       recorded_sha256=None, **audit.signature(path)))
    removed = {ROOT / item['path'] for item in candidates}
    assert not keep.intersection(removed)
    assert not any(referenced(path, local_references) for path in removed)
    assert len([path for path in epochs if path not in removed]) >= min(10, len(epochs))
    retained = {}
    for path in run.rglob('*'):
        if path.is_file() and not path.is_symlink() and path not in removed and path.name not in {'.training.lock', '.cleanup.lock'}:
            record = audit.signature(path)
            if record['size'] <= 5 * 1024**2:
                record['sha256'] = audit.sha256(path)
            retained[str(path.relative_to(ROOT))] = record
    return dict(run=str(run.relative_to(ROOT)), latest=str(latest.relative_to(ROOT)),
                configuration_sha256=audit.sha256(run / 'run_config.json'),
                keep_epochs=[str(path.relative_to(ROOT)) for path in sorted(keep) if path.is_file()],
                candidates=candidates, retained=retained)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert not (HERE / 'execution_result.json').exists(), 'This cleanup already completed'
    queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
    assert 'No unfinished job found' in queue, 'Jobs active; stop cleanup'
    runs, references, errors = inventory()
    with ExitStack() as stack:
        plans, skipped = [], []
        for run in runs:
            if not (run / 'TRAIN_COMPLETE.json').is_file():
                skipped.append(dict(run=str(run.relative_to(ROOT)), reason='no completed-training marker'))
                continue
            if not (run / '.training.lock').exists():
                skipped.append(dict(run=str(run.relative_to(ROOT)), reason='trainer locking contract not established'))
                continue
            lock = stack.enter_context((run / '.training.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                plan = plan_run(run, references, errors)
            except (AssertionError, KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
                skipped.append(dict(run=str(run.relative_to(ROOT)), reason=f'{type(error).__name__}: {error}'))
                continue
            plans.append(plan)
        candidates = [item for plan in plans for item in plan['candidates']]
        assert all(item['links'] == 1 for item in candidates), 'Shared candidate inode requires separate reference audit'
        allocated = sum(item['allocated_bytes'] for item in candidates)
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), qstat=queue,
            scope='Repository recipes dated 2026-10-13 through 2026-10-20; only completed compatible Trainer runs. BPE500 and previously cleaned POWSM excluded.',
            policy='Retain first/best/newest-ten epochs, all JSON-referenced weights and checkpoint directories, latest validated full resume state, average models and all noncandidate files. Unknown, incomplete and locked runs skipped.',
            plans=plans, skipped=skipped, reference_index={str(path.relative_to(ROOT)): sorted(records) for path, records in references.items()},
            reference_parse_errors=errors, candidates=len(candidates), estimated_reclaim_bytes=allocated)
        audit.write_json(HERE / ('approved_manifest.json' if args.execute else 'plan.json'), manifest)
        print(json.dumps(dict(compatible_runs=len(plans), files=len(candidates), estimated_GiB=allocated / 2**30,
            per_run=[dict(run=plan['run'], files=len(plan['candidates']), GiB=sum(item['allocated_bytes'] for item in plan['candidates']) / 2**30) for plan in plans],
            skipped=skipped, reference_parse_errors=len(errors)), indent=2), flush=True)
        if not args.execute:
            return
        deleted = 0
        with (HERE / 'unlink_journal.jsonl').open('a') as journal:
            for item in candidates:
                path = ROOT / item['path']
                assert not path.is_symlink() and path.is_relative_to(REPRODUCTION)
                current = audit.signature(path)
                assert current == {key: item[key] for key in current}, str(path)
                digest = item['recorded_sha256'] or audit.sha256(path)
                journal.write(json.dumps(dict(event='before_unlink', path=item['path'], sha256=digest,
                    hash_kind='existing epoch metadata' if item['recorded_sha256'] else 'computed before deletion')) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                path.unlink()
                deleted += 1
                journal.write(json.dumps(dict(event='unlinked', path=item['path'])) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                if deleted % 10 == 0:
                    print(f'Deleted {deleted}/{len(candidates)} audited payloads', flush=True)
        retained_count = 0
        for plan in plans:
            for relative, original in plan['retained'].items():
                path = ROOT / relative
                current = audit.signature(path)
                assert current == {key: original[key] for key in current}, relative
                if 'sha256' in original:
                    assert audit.sha256(path) == original['sha256'], relative
                retained_count += 1
        assert all(not (ROOT / item['path']).exists() for item in candidates)
        quota = subprocess.run(['lfs', 'quota', '-g', 'gm64', '/work'], capture_output=True, text=True, check=True).stdout
        result = dict(deleted_files=deleted, verified_retained_files=retained_count,
                      estimated_reclaim_bytes=allocated, quota_after=quota, evaluation_uploads=False,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        audit.write_json(HERE / 'execution_result.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()