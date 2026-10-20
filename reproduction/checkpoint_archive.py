"""Preserve model-only checkpoint snapshots before Trainer rotates resumable files."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'checkpoint_comparison'


def archive(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('checkpoint-*')):
        weights = path/'pytorch_model.bin'
        state = path/'trainer_state.json'
        if not weights.exists() or not state.exists():
            continue
        target = destination/path.name
        if (target/'ARCHIVED.json').exists():
            continue
        target.mkdir(exist_ok=True)
        try:
            record = json.loads(state.read_text())
            for name in ['pytorch_model.bin', 'config.json', 'trainer_state.json']:
                original, saved = path/name, target/name
                if saved.exists():
                    continue
                if name == 'pytorch_model.bin':
                    try:
                        os.link(original, saved)
                    except OSError:
                        shutil.copy2(original, saved)
                else:
                    shutil.copy2(original, saved)
            (target/'ARCHIVED.json').write_text(json.dumps({
                'source': str(path.resolve()), 'step': record['global_step'],
                'archived_at': time.time(), 'model_only': True,
                'original_checkpoint_unchanged': True})+'\n')
        except (FileNotFoundError, json.JSONDecodeError):
            # A live save/rotation may race the watcher; retry completed saves.
            continue


def archive_training(output):
    output = Path(output)
    dataset = output.parents[1].name.removeprefix('multihead_')
    archive(output, ROOT/(dataset+'_'+output.name)/'snapshots')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-libri100', action='store_true')
    args = parser.parse_args()
    production = Path(__file__).resolve().parent/'multihead_libri100/models'
    while True:
        for objective in ['ctc', 'ce', 'jsd_upstream']:
            archive_training(production/objective)
        if not args.watch_libri100 or all((production/o/'TRAIN_COMPLETE.json').exists()
                                          for o in ['ctc', 'ce', 'jsd_upstream']):
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
