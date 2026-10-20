"""Read the existing TIMIT40 phone-recognition manifests, without rewriting them."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'reproduction/multihead_timit/data'
VOCAB = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(train_limit=0, eval_limit=0):
    vocab = json.loads(VOCAB.read_text())
    if set(vocab.values()) != set(range(42)) or vocab['<pad>'] != 0:
        raise ValueError('Expected the frozen 42-ID single-head TIMIT40 inventory')
    rows, hashes, seen = {}, {}, set()
    for split, name, expected in [('train', 'train-train', 4620), ('dev', 'train-dev', 400), ('test', 'test', 192)]:
        path = DATA / (name + '.jsonl')
        source = [json.loads(line) for line in path.read_text().splitlines()]
        if len(source) != expected:
            raise ValueError(f'Unexpected {split} size: {len(source)}')
        ids = [r['utt'] for r in source]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping utterance IDs')
        seen.update(ids)
        for row in source:
            if [vocab[p] for p, _, _ in row['references']['ottc_arpabet']] != row['targets']['ottc_arpabet']:
                raise ValueError('Manifest inventory differs from the frozen single-head vocabulary')
        hashes[split] = sha(path)
        limit = train_limit if split == 'train' else eval_limit
        rows[split] = source[:limit] if limit else source
    return rows, vocab, dict(manifests_sha256=hashes, vocab_sha256=sha(VOCAB))


def read_audio(row):
    path = row['wav'].replace('/home/m64000/work/', '/work/gm64/m64000/')
    audio, sr = sf.read(path, dtype='float32')
    if sr != 16000 or audio.ndim != 1 or not len(audio):
        raise ValueError(f'Expected nonempty 16kHz mono audio: {path}')
    return audio


def collate(rows, device):
    # Same utterance normalization as Wav2Vec2FeatureExtractor(do_normalize=True).
    audio = [read_audio(row) for row in rows]
    lengths = [len(a) for a in audio]
    values = torch.zeros(len(rows), max(lengths))
    mask = torch.zeros_like(values, dtype=torch.long)
    labels = torch.full((len(rows), max(len(r['targets']['ottc_arpabet']) for r in rows)), -100, dtype=torch.long)
    for index, (wave, row) in enumerate(zip(audio, rows)):
        wave = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)
        values[index, :len(wave)] = torch.from_numpy(wave)
        mask[index, :len(wave)] = 1
        tokens = row['targets']['ottc_arpabet']
        labels[index, :len(tokens)] = torch.tensor(tokens)
    return dict(input_values=values.to(device), attention_mask=mask.to(device), labels=labels.to(device))
