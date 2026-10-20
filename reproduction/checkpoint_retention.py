"""Bound epoch archives after a verified Trainer save, without changing training."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import zipfile

from transformers import TrainerCallback

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'data', 'datasets', 'vendor', 'runtime', 'conda_pkgs', 'cache', 'pip_cache',
             'source_snapshot', 'source_snapshots', 'epoch_weights', 'wandb', '__pycache__',
             'runtime_retention', 'runtime_batching', 'snapshots', 'import_source'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def references(root):
    found = set()
    for directory, children, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        children[:] = [name for name in children if name not in SKIP_DIRS and
                       not name.startswith(('.', 'checkpoint-')) and not (parent / name).is_symlink()]
        for filename in filenames:
            path = parent / filename
            if path.suffix != '.json' or path.is_symlink() or path.stat().st_size > 16 * 1024**2:
                continue
            record = json.loads(path.read_text())
            for value in strings(record):
                if not re.search(r'\.pt$|/epoch_weights/?$', value):
                    continue
                target = Path(value)
                if not target.is_absolute():
                    target = (ROOT / target) if value.startswith('reproduction/') else (parent / target)
                found.add(target.resolve())
    return found


def verify_saved_checkpoint(output, step):
    checkpoint = output / f'checkpoint-{step}'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == step
    model = checkpoint / 'model.safetensors'
    with model.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < length < 16 * 1024**2
        header = json.loads(stream.read(length))
    offsets = [value['data_offsets'] for key, value in header.items() if key != '__metadata__']
    assert offsets and max(end for _, end in offsets) + 8 + length == model.stat().st_size
    for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin']:
        with zipfile.ZipFile(checkpoint / filename) as archive:
            assert archive.namelist()
    if (checkpoint / 'COMPLETE.json').exists():
        marker = json.loads((checkpoint / 'COMPLETE.json').read_text())
        assert marker['step'] == step
        assert all((checkpoint / name).stat().st_size == size for name, size in marker['files'].items())


def prune_epochs(output, step, reference_root=None, keep_last=10, keep_best=3):
    output = Path(output).resolve()
    assert keep_last >= 10 and keep_best >= 1
    members = sorted((output / 'epoch_weights').glob('epoch-*.pt'))
    if len(members) <= keep_last:
        return dict(removed=[], kept=len(members))
    try:
        verify_saved_checkpoint(output, step)
        metadata = {path: json.loads(path.with_suffix('.json').read_text()) for path in members}
        assert all(record['step'] <= step and record.get('weights_sha256') for record in metadata.values())
        scores = {}
        for line in (output / 'history.jsonl').open():
            row = json.loads(line)
            if 'eval_WER_percent' not in row or row.get('optimizer_step', step + 1) > step:
                continue
            epoch = round(row['epoch'])
            if abs(row['epoch'] - epoch) < 1e-6:
                scores[epoch] = row['eval_WER_percent']
        available_epochs = {record['epoch'] for record in metadata.values()}
        scores = {epoch: score for epoch, score in scores.items() if epoch in available_epochs}
        assert len(scores) >= keep_best, 'insufficient dev selection evidence'
        winners = {epoch for epoch, _ in sorted(scores.items(), key=lambda item: (item[1], item[0]))[:keep_best]}
        best = json.loads((output / 'BEST_DEV.json').read_text())
        winners.add(round(best['epoch']))
        pins = references(reference_root or ROOT / 'reproduction')
        keep = set(members[-keep_last:])
        for path, record in metadata.items():
            if record['epoch'] in winners or path in pins or path.parent in pins or path.is_symlink():
                keep.add(path)
        candidates = [path for path in members if path not in keep]
    except (AssertionError, KeyError, ValueError, OSError, struct.error, zipfile.BadZipFile) as error:
        return dict(removed=[], skipped=f'{type(error).__name__}: {error}')
    journal_dir = output / 'runtime_retention'
    journal_dir.mkdir(exist_ok=True)
    removed = []
    with (journal_dir / 'unlink_journal.jsonl').open('a') as journal:
        for path in candidates:
            info = path.stat()
            record = dict(path=str(path), epoch=metadata[path]['epoch'], saved_step=step,
                recorded_sha256=metadata[path]['weights_sha256'], bytes=info.st_size,
                timestamp_utc=datetime.now(timezone.utc).isoformat())
            journal.write(json.dumps(dict(event='before_unlink', **record)) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            assert path.stat().st_mtime_ns == info.st_mtime_ns and path.stat().st_size == info.st_size
            path.unlink()
            removed.append(str(path))
            journal.write(json.dumps(dict(event='unlinked', **record)) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
    return dict(removed=removed, kept=len(keep), best_epochs=sorted(winners))


class CheckpointRetentionCallback(TrainerCallback):
    def __init__(self, output):
        self.output = Path(output)
        directory = self.output / 'runtime_retention'
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(__file__)
        digest = sha256(source)
        snapshot = directory / (digest + '.py')
        if not snapshot.exists():
            snapshot.write_bytes(source.read_bytes())
        self.policy = dict(keep_last_epochs=10, keep_best_dev_epochs=3, metric='eval_WER_percent',
            preserve_best_file=True, preserve_detected_json_references=True,
            full_resume_checkpoints='Existing trainer retains latest two; this callback never deletes them.',
            source_sha256=digest, evaluation_uploads=False)
        (directory / 'policy.json').write_text(json.dumps(self.policy, indent=2) + '\n')

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        result = prune_epochs(self.output, state.global_step)
        if result.get('skipped') or result['removed']:
            print('Checkpoint retention: ' + json.dumps(result), flush=True)
        return control


def check():
    import torch
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / 'run'
        epochs = output / 'epoch_weights'
        epochs.mkdir(parents=True)
        history = []
        for epoch in range(1, 21):
            path = epochs / f'epoch-{epoch:03d}.pt'
            torch.save(dict(model=torch.tensor([epoch])), path)
            path.with_suffix('.json').write_text(json.dumps(dict(epoch=epoch, step=epoch * 10, weights_sha256=sha256(path))))
            history.append(dict(epoch=epoch, optimizer_step=epoch * 10, eval_WER_percent={2: 3, 5: 1, 8: 2}.get(epoch, 99)))
        (output / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
        (output / 'BEST_DEV.json').write_text(json.dumps(dict(epoch=5, step=50)))
        torch.save(dict(model=torch.tensor([5])), output / 'best_dev.pt')
        (root / 'external_evaluation.json').write_text(json.dumps(dict(checkpoint=str(epochs / 'epoch-004.pt'))))
        checkpoint = output / 'checkpoint-200'
        checkpoint.mkdir()
        (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=200)))
        for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin']:
            torch.save(dict(value=torch.tensor([1])), checkpoint / filename)
        (checkpoint / 'model.safetensors').write_bytes(b'incomplete')
        assert prune_epochs(output, 200, root).get('skipped')
        assert len(list(epochs.glob('*.pt'))) == 20
        save_file({'weight': torch.ones(1)}, str(checkpoint / 'model.safetensors'))
        result = prune_epochs(output, 200, root)
        assert len(result['removed']) == 6, result
        assert {int(path.stem.split('-')[1]) for path in epochs.glob('*.pt')} == {2, 4, 5, 8, *range(11, 21)}
        assert len(list(epochs.glob('*.json'))) == 20
        assert (output / 'best_dev.pt').is_file()
        assert all((checkpoint / name).is_file() for name in ['model.safetensors', 'optimizer.pt', 'rng_state.pth'])
        assert not prune_epochs(output, 200, root)['removed']
    print('PASS: best-three/latest-ten/reference preservation, incomplete-save rejection, metadata preservation and idempotence')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    check()