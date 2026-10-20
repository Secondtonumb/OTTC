"""Freeze POWSM's native text tokenizer and encode existing TIMIT splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata
import urllib.request

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = '21ffa410432ace159f3c1fdeed304c70eddf7d34'
MODEL_FILE = 'data/token_list/bpe_unigram40000/bpe.model'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    temporary.replace(path)


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('\u2019', '').replace("'", '')
    return ' '.join(re.findall(r'[a-z]+', text))


class Tokenizer:
    def __init__(self, directory):
        self.sp = spm.SentencePieceProcessor(model_file=str(directory / 'bpe.model'))
        self.tokens = json.loads((directory / 'tokens.json').read_text())
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        assert len(self.tokens) == len(self.token_to_id) == 40002
        assert self.tokens[:2] == ['<blank>', '<unk>']
        self.blank_id, self.unk_id = 0, 1

    def encode(self, text):
        pieces = self.sp.encode(text, out_type=str)
        return [self.token_to_id.get(piece, self.unk_id) for piece in pieces]

    def decode(self, ids):
        pieces = [self.tokens[int(index)] for index in ids if int(index) >= 0 and int(index) != self.blank_id]
        return self.sp.decode(pieces)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'tokenizer'
    assets.mkdir(exist_ok=True)
    model_path = assets / 'bpe.model'
    url = f'https://huggingface.co/espnet/powsm/resolve/{REVISION}/{MODEL_FILE}'
    if not model_path.exists():
        temporary = model_path.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(model_path)
    config_path = ROOT / 'audits/20261008/phone_inventory_sources/powsm_config.yaml'
    config = yaml.safe_load(config_path.read_text())
    assert config['token_type'] == 'bpe' and config['bpemodel'] == MODEL_FILE
    write_json(assets / 'tokens.json', config['token_list'])
    tokenizer = Tokenizer(assets)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    piece_types = {piece.piece: piece.type for piece in proto.pieces}
    special = {piece for piece, kind in piece_types.items() if kind != 1}
    aliases = {'<s>': '<sos>', '</s>': '<eos>'}
    mapping = [tokenizer.token_to_id.get(aliases.get(tokenizer.sp.id_to_piece(index), tokenizer.sp.id_to_piece(index)))
               for index in range(tokenizer.sp.vocab_size())]
    assert all(index is not None for index in mapping), 'SentencePiece and ESPnet vocabularies differ'
    write_json(assets / 'sentencepiece_to_output_ids.json', mapping)
    source_dir = ROOT / 'reproduction/multihead_timit/data'
    statistics, source_hashes, seen_ids, speakers, split_tokens = {}, {}, {}, {}, {}
    for split, filename, expected in [('train', 'train-train', 4620), ('dev', 'train-dev', 400), ('test', 'test', 192)]:
        source = source_dir / (filename + '.jsonl')
        original = [json.loads(line) for line in source.open()]
        assert len(original) == expected
        rows, counts, lengths, examples = [], Counter(), [], []
        seen_ids[split], speakers[split], split_tokens[split] = set(), set(), set()
        for row in original:
            text = normalize_text(row['text'])
            assert text, row['utt']
            labels = tokenizer.encode(text)
            assert labels and tokenizer.unk_id not in labels, (row['utt'], text)
            assert all(tokenizer.tokens[index] not in special for index in labels), row['utt']
            assert tokenizer.decode(labels) == text, (row['utt'], text, tokenizer.decode(labels))
            assert row['utt'] not in seen_ids[split]
            assert Path(row['wav']).is_file(), row['wav']
            seen_ids[split].add(row['utt'])
            speakers[split].add(row['utt'].split('_')[0])
            split_tokens[split].update(labels)
            counts.update(labels)
            lengths.append(len(labels))
            rows.append(dict(utt=row['utt'], wav=row['wav'], duration=row['duration'],
                             original_text=row['text'], text=text, labels=labels))
            if len(examples) < 3:
                examples.append(dict(text=text, pieces=[tokenizer.tokens[index] for index in labels], ids=labels))
        temporary = output / (split + '.jsonl.tmp')
        with temporary.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + '\n')
        temporary.replace(output / (split + '.jsonl'))
        source_hashes[split] = sha256(source)
        statistics[split] = dict(utterances=len(rows), speakers=len(speakers[split]),
            unknown_tokens=0, roundtrip_failures=0, tokens=sum(lengths), unique_tokens=len(counts),
            mean_tokens=sum(lengths) / len(lengths), max_tokens=max(lengths), examples=examples)
    for first, second in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not seen_ids[first] & seen_ids[second], (first, second, 'utterance overlap')
        assert not speakers[first] & speakers[second], (first, second, 'speaker overlap')
    for split in ['dev', 'test']:
        unseen = split_tokens[split] - split_tokens['train']
        statistics[split]['token_types_absent_from_timit_train'] = len(unseen)
    metadata = dict(tokenizer_repo='espnet/powsm', revision=REVISION, url=url,
        algorithm=sentencepiece_model_pb2.TrainerSpec.ModelType.Name(proto.trainer_spec.model_type),
        sentencepiece_version=spm.__version__, sentencepiece_vocab_size=tokenizer.sp.vocab_size(),
        output_vocab_size=len(tokenizer.tokens), tokenizer_sha256=sha256(model_path),
        tokens_sha256=sha256(assets / 'tokens.json'), source_config_sha256=sha256(config_path),
        prepare_source_sha256=sha256(Path(__file__)), source_manifests_sha256=source_hashes,
        manifests_sha256={split: sha256(output / (split + '.jsonl')) for split in statistics},
        target='TIMIT orthographic text, not phones; native POWSM segmentation and full ESPnet output IDs',
        normalization="NFKC, lowercase, delete apostrophes (dont); other punctuation separates ASCII letter words",
        normalization_adaptation='Original POWSM tokenizer has no apostrophe token; scoring uses the same explicit normalization',
        task_prefix=False, timestamp_tokens=False, tokenizer_retrained=False,
        output_vocab_pruned=False, split_policy='existing speaker-disjoint 4620 train / 400 dev / 192 core test; train includes SA',
        evaluation_uploads=False, statistics=statistics)
    write_json(output / 'metadata.json', metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'data')
    prepare(parser.parse_args().output)