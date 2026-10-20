"""Read portable full-corpus split manifests, retaining legacy pilot support."""
import gzip
import hashlib
import json
from pathlib import Path


def load_split(path, split=None):
    path = Path(path)
    meta = json.loads(path.read_text())
    split = split or meta.get('train_split', 'train-clean-100')
    if 'split_manifests' in meta:
        entry = meta['split_manifests'][split]
        source = path.parent / entry['path']
        h = hashlib.sha256()
        with source.open('rb') as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('Split manifest checksum mismatch: ' + str(source))
        with gzip.open(source, 'rt') as f:
            rows = [json.loads(line) for line in f]
        if len(rows) != entry['count']:
            raise ValueError('Split manifest row count mismatch')
    else:
        rows = meta['splits'][split]
    return meta, rows
