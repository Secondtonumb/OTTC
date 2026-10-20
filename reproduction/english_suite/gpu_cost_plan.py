"""Snapshot existing stable LibriSpeech checkpoints for an isolated compute benchmark."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/work/gm64/m64000/OTTC')


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    plan = {'job': os.environ.get('PBS_JOBID'), 'production_updated': False, 'checkpoints': {}}
    for objective in ('ctc', 'ce', 'jsd_upstream'):
        directory = ROOT/'reproduction/checkpoint_comparison'/f'libri100_{objective}'/'snapshots'
        candidates = [p for p in directory.glob('checkpoint-*')
                      if (p/'config.json').is_file() and (p/'pytorch_model.bin').is_file()]
        source = max(candidates, key=lambda p: int(p.name.split('-')[1]))
        target = args.output/'checkpoints'/objective
        target.mkdir(parents=True)
        hashes = {}
        for name in ('config.json', 'pytorch_model.bin'):
            shutil.copy2(source/name, target/name)
            hashes[name] = sha256(target/name)
            assert hashes[name] == sha256(source/name)
        plan['checkpoints'][objective] = {'source': str(source), 'snapshot': str(target), 'sha256': hashes}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2) + '\n')


if __name__ == '__main__':
    main()
