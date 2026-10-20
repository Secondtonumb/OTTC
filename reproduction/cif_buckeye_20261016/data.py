"""Reuse original-human Buckeye partial19 and TIMIT40 IDs for CIF occurrences."""
import json
from pathlib import Path

from reproduction.cif_timit_20261016.data import ROOT, VOCAB, collate, read_audio, sha

DATA = ROOT / 'reproduction/buckeye_singlehead_20261014/data'


def load_rows(train_limit=0, eval_limit=0):
    vocab = json.loads(VOCAB.read_text())
    if set(vocab.values()) != set(range(42)) or vocab['<pad>'] != 0:
        raise ValueError('Expected frozen 42-ID TIMIT40 inventory')
    rows, hashes, speakers, utterances, tracks = {}, {}, {}, {}, {}
    for split, expected in [('train', 10927), ('dev', 1524), ('test', 1840)]:
        path = DATA / (split + '.jsonl')
        source = [json.loads(line) for line in path.read_text().splitlines()]
        if len(source) != expected:
            raise ValueError(f'Unexpected {split} size: {len(source)}')
        hashes[split] = sha(path)
        speakers[split] = {r['speaker'] for r in source}
        utterances[split] = {r['utt'] for r in source}
        tracks[split] = {r['source_utt'] for r in source}
        if len(utterances[split]) != len(source):
            raise ValueError('Duplicate utterances')
        adapted = []
        for row in source:
            # Occurrences, rather than CTC-collapsed phones or synthetic edge sil.
            reference = row['reference']
            ids = [vocab[p] for p, _, _ in reference]
            if not ids or any(i in (0, 41) for i in ids):
                raise ValueError('Empty or unknown human phone sequence')
            frames = (row['samples'] - 400) // 320 + 1
            if len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > frames:
                raise ValueError('CTC-infeasible human phone sequence')
            if not all(0 <= s < e <= row['duration'] + 1e-4 for _, s, e in reference):
                raise ValueError('Invalid human interval')
            adapted.append(dict(row, references={'ottc_arpabet': reference},
                                targets={'ottc_arpabet': ids}))
        limit = train_limit if split == 'train' else eval_limit
        rows[split] = adapted[:limit] if limit else adapted
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        if speakers[a] & speakers[b] or utterances[a] & utterances[b] or tracks[a] & tracks[b]:
            raise ValueError('Buckeye split leakage')
    return rows, vocab, dict(manifests_sha256=hashes, vocab_sha256=sha(VOCAB),
        adapter_sha256=sha(Path(__file__)), shared_data_sha256=sha(ROOT / 'reproduction/cif_timit_20261016/data.py'),
        corpus='Buckeye original-human partial19, speaker-disjoint 15/2/2',
        phone_mapping='ao->aa from existing source; repeated human occurrences retained; no artificial edge silence')
