"""Preserve a complete global64 checkpoint for single-GPU local64 continuation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.libri100_bpe500_jsd_20261021 import train as previous

parent = previous.parent


def snapshot(source, destination):
    ready = destination / 'IMPORT_COMPLETE.json'
    if ready.exists():
        return parent.read_json(ready)
    config = parent.read_json(source / 'run_config.json')
    assert config['local_batch'] == 32 and config['global_batch'] == 64 and config['world_size'] == 2
    assert config['epochs'] == 50 and config['planned_optimizer_steps'] == 22300
    candidates = []
    for path in source.glob('checkpoint-*'):
        if parent.valid_checkpoint(path):
            state = parent.read_json(path / 'trainer_state.json')
            candidates.append((path, state))
    assert candidates, f'No complete checkpoint: {source}'
    boundaries = [item for item in candidates if abs(item[1]['epoch'] - round(item[1]['epoch'])) < 1e-6]
    checkpoint, state = max(boundaries or candidates, key=lambda item: item[1]['global_step'])
    assert state['global_step'] < config['planned_optimizer_steps'], 'Do not restart a completed experiment'
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), f'Inspect partial destination: {destination}'
    temporary = Path(tempfile.mkdtemp(prefix='.local64-import-', dir=destination.parent))
    try:
        target = temporary / checkpoint.name
        target.mkdir()
        marker = parent.read_json(checkpoint / 'COMPLETE.json')
        for filename in marker['files']:
            if filename == 'trainer_state.json':
                continue
            os.link(checkpoint / filename, target / filename)
        assert previous.prepare.sha256(target / 'model.safetensors') == marker['model_sha256']
        for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth']:
            with zipfile.ZipFile(target / filename) as archive:
                assert archive.namelist()
        os.link(target / 'rng_state_0.pth', target / 'rng_state.pth')
        parent.write_json(target / 'SOURCE_TRAINER_STATE.json', state)
        parent.write_json(target / 'trainer_state.json', dict(state, train_batch_size=64))
        parent.write_json(target / 'SOURCE_COMPLETE.json', marker)
        files = list(marker['files']) + ['rng_state.pth']
        parent.write_json(target / 'COMPLETE.json', dict(marker, world_size=1, source_world_size=2,
            files={filename: (target / filename).stat().st_size for filename in files},
            adaptation='Global batch/step/epoch/optimizer/scheduler unchanged; local32x2 -> local64x1; rank0 RNG selected.'))
        history = []
        with (source / 'history.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get('optimizer_step', state['global_step'] + 1) <= state['global_step']:
                    history.append(record)
        (temporary / 'history.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in history))
        archive = temporary / 'epoch_weights'
        archive.mkdir()
        for metadata_path in (source / 'epoch_weights').glob('epoch-*.json'):
            metadata = parent.read_json(metadata_path)
            if metadata['step'] <= state['global_step']:
                weights = metadata_path.with_suffix('.pt')
                if weights.exists():
                    os.link(weights, archive / weights.name)
                    shutil.copy2(metadata_path, archive / metadata_path.name)
        eligible = [record for record in history if 'eval_WER_percent' in record and
                    abs(record['epoch'] - round(record['epoch'])) < 1e-6 and
                    (archive / f"epoch-{round(record['epoch']):03d}.pt").exists()]
        assert eligible, 'No completed epoch available for best-dev preservation'
        best = min(eligible, key=lambda record: record['eval_WER_percent'])
        best_path = archive / f"epoch-{round(best['epoch']):03d}.pt"
        os.link(best_path, temporary / 'best_dev.pt')
        parent.write_json(temporary / 'BEST_DEV.json', dict(epoch=best['epoch'], step=best['optimizer_step'],
            WER_percent=best['eval_WER_percent'], checkpoint=str(destination / 'best_dev.pt'),
            weights_sha256=previous.prepare.sha256(best_path), selection='Inherited completed-epoch dev WER only'))
        parent.write_json(temporary / 'SOURCE_RUN_CONFIG.json', config)
        record = dict(source=str(source), checkpoint=str(checkpoint), checkpoint_name=checkpoint.name,
            step=state['global_step'], epoch=state['epoch'], epoch_boundary=bool(boundaries),
            original_config_sha256=previous.prepare.sha256(source / 'run_config.json'),
            model_sha256=marker['model_sha256'], optimizer_sha256=previous.prepare.sha256(target / 'optimizer.pt'),
            scheduler_sha256=previous.prepare.sha256(target / 'scheduler.pt'),
            old_world_size=2, new_world_size=1, old_local_batch=32, new_local_batch=64, global_batch=64,
            optimizer_schedule_unchanged=True, original_artifacts_untouched=True,
            sampling_note='World size and local batch alter length-grouped sample order; epoch-boundary checkpoint preferred. Not bitwise-equivalent replay.',
            evaluation_uploads=False)
        parent.write_json(temporary / 'IMPORT_COMPLETE.json', record)
        temporary.rename(destination)
        print(json.dumps(record, indent=2), flush=True)
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def check():
    import torch
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as directory:
        source, target = Path(directory) / 'source', Path(directory) / 'target'
        checkpoint = source / 'checkpoint-446'
        checkpoint.mkdir(parents=True)
        parent.write_json(source / 'run_config.json', dict(local_batch=32, global_batch=64,
            world_size=2, epochs=50, planned_optimizer_steps=22300))
        state = dict(global_step=446, epoch=1., max_steps=22300, train_batch_size=32)
        parent.write_json(checkpoint / 'trainer_state.json', state)
        parent.write_json(checkpoint / 'config.json', {})
        save_file({'weight': torch.ones(2)}, str(checkpoint / 'model.safetensors'))
        for filename in ['optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth', 'training_args.bin']:
            torch.save({'step': 446}, checkpoint / filename)
        files = [path.name for path in checkpoint.iterdir()]
        parent.write_json(checkpoint / 'COMPLETE.json', dict(step=446, world_size=2,
            model_sha256=previous.prepare.sha256(checkpoint / 'model.safetensors'),
            files={name: (checkpoint / name).stat().st_size for name in files}))
        archive = source / 'epoch_weights'
        archive.mkdir()
        torch.save(dict(model={'weight': torch.ones(2)}), archive / 'epoch-001.pt')
        parent.write_json(archive / 'epoch-001.json', dict(epoch=1, step=446,
            weights_sha256=previous.prepare.sha256(archive / 'epoch-001.pt')))
        (source / 'history.jsonl').write_text(json.dumps(dict(epoch=1., optimizer_step=446, eval_WER_percent=90.)) + '\n')
        result = snapshot(source, target)
        assert result['epoch_boundary'] and result['step'] == 446
        assert parent.read_json(target / 'checkpoint-446/trainer_state.json') == dict(state, train_batch_size=64)
        assert parent.read_json(checkpoint / 'trainer_state.json') == state
        assert parent.valid_checkpoint(target / 'checkpoint-446')
        assert previous.prepare.sha256(checkpoint / 'optimizer.pt') == result['optimizer_sha256']
        assert snapshot(source, target) == result
    print('PASS: immutable import, rank0 RNG adaptation, unchanged steps/schedule and source preservation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        for recipe in previous.RECIPES:
            snapshot(previous.HERE / 'models' / recipe, HERE / 'models' / recipe)