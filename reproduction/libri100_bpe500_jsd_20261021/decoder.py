"""Audio-only S3T2** decoding using the existing frozen token-loop graph."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.timit_bpe500_stages_20261020 import train as parent
from reproduction.native_stride_ottc_20261018.graphs import graph, decode
from reproduction.libri100_bpe500_jsd_20261021.prepare import DATA, Tokenizer

torch = parent.torch


def build(tokenizer, device):
    topology = parent.base.build_topology(tokenizer, 'spottc_s3t2')
    from utils.subphonetic_topology import token_fst_text
    phones = [f'bpe{index:05d}' for index in range(4, len(tokenizer.tokens))]
    enriched = dict(topology, phones=phones, token_fst_text=token_fst_text)
    vector = graph(enriched, {phone: int(phone.removeprefix('bpe')) for phone in phones}, device=device)
    return topology, vector


def best_path(logits, topology, vector):
    tokens, path, starts = decode(logits.float(), vector)
    assert len(tokens) == len(starts)
    for index, token in enumerate(tokens):
        finish = starts[index + 1] if index + 1 < len(starts) else len(path)
        labels = [label for label in path[starts[index]:finish] if label != 0]
        states = topology['mapping'][token]
        assert labels[0] == states[0] and labels[-1] == states[2]
        sequence = [states.index(label) for label in labels]
        assert sequence == sorted(sequence), 'Nonmonotonic states inside one token occurrence'
    return tokens, path, starts


def check():
    tokenizer = Tokenizer(DATA / 'tokenizer')
    topology, vector = build(tokenizer, 'cpu')
    token = tokenizer.encode('himself')
    assert len(token) == 1
    token = token[0]
    other = tokenizer.sp.piece_to_id('self')
    states = topology['mapping'][token]
    scores = torch.full((8, len(topology['labels'])), -20.)
    scores[:, 0] = -4
    for frame, state in enumerate([0, 0, 1, 1, 1, 2, 2, 2]):
        scores[frame, states[state]] = 10
    scores[4, topology['mapping'][other][1]] = 11
    tokens, path, _ = best_path(scores, topology, vector)
    assert tokens == [token]
    assert topology['mapping'][other][1] not in path
    scores = torch.full((2, len(topology['labels'])), -20.)
    scores[:, 0] = -4
    scores[0, states[0]], scores[1, states[2]] = 10, 10
    assert best_path(scores, topology, vector)[0] == [token]
    scores = torch.full((1, len(topology['labels'])), -20.)
    scores[0, 0], scores[0, states[1]] = 0, 10
    assert best_path(scores, topology, vector)[0] == []
    print('PASS: S3T2** keeps one complete token across an isolated foreign S1; S0/S2 mandatory, S1 optional')


if __name__ == '__main__':
    check()