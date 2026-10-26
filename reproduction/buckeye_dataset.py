"""Default Buckeye evaluation source for newly created recipes (local artifacts)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / 'reproduction/dataset_defaults.json'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_buckeye():
    policy = json.loads(DEFAULTS.read_text())['evaluation']['buckeye']
    data = ROOT / policy['data_root']
    complete = json.loads((data / 'PREPARE_COMPLETE.json').read_text())
    if not complete['validated'] or complete['dataset_id'] != policy['dataset_id']:
        raise ValueError('Buckeye default is not the validated full40 dataset')
    if _sha(data / 'PREPARE_COMPLETE.json') != policy['prepare_complete_sha256']:
        raise ValueError('Buckeye validation contract changed; revalidate and activate explicitly')
    for name in ['audit.json', 'inventory_mapping.json', 'vocab.json']:
        if _sha(data/name) != complete['metadata_sha256'][name]:
            raise ValueError('Buckeye metadata changed: ' + name)
    return data, complete


def load_buckeye_rows(split='test', *, model_vocab=None, training_speakers=()):
    """Return historical-schema rows plus a checked full40 data contract.

    Pass known Buckeye training speakers when evaluating a corpus-trained model;
    overlapping speakers raise an error rather than silently changing the subset.
    """
    if split not in ['train', 'dev', 'test']:
        raise ValueError('Choose train, dev or test explicitly')
    data, complete = resolve_buckeye()
    path = data / (split + '.jsonl')
    if _sha(path) != complete['manifests_sha256'][split]:
        raise ValueError('Buckeye manifest changed: ' + str(path))
    vocab = json.loads((data/'vocab.json').read_text())
    if model_vocab is not None and model_vocab != vocab:
        raise ValueError('Model inventory/IDs differ from ottc_legacy42; use an explicit model-specific adapter')
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    overlap = set(training_speakers) & {r['speaker'] for r in rows}
    if overlap:
        raise ValueError('Buckeye evaluation speakers overlap model training speakers: ' + ', '.join(sorted(overlap)))
    contract = dict(dataset_id=complete['dataset_id'], data_root=str(data), split=split,
                    manifest_sha256=complete['manifests_sha256'][split],
                    source_sha256=complete['source_sha256'],
                    vocab_sha256=complete['metadata_sha256']['vocab.json'],
                    utterances=len(rows), inventory='ottc_legacy42',
                    reference_times_used_by_aligner=False, evaluation_uploads=False)
    return rows, vocab, contract
