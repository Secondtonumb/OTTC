"""Inspect the public tokenizer and train a true LibriSpeech BPE500 model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2


HERE = Path(__file__).resolve().parent
REPOSITORY = 'csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11'
REVISION = 'ee9731ed5e6941ad58716704159d08cf46d34f35'


def inspect_public():
    assets = HERE / 'tokenizer'
    assets.mkdir(parents=True, exist_ok=True)
    sources = {}
    for filename in ['bpe.model', 'tokens.txt']:
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/data/lang_bpe_500/{filename}'
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        destination = assets / filename
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError(f'Refusing to replace changed asset: {destination}')
        else:
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(destination)
        sources[filename] = dict(url=url, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))

    tokenizer = spm.SentencePieceProcessor(model_file=str(assets / 'bpe.model'))
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((assets / 'bpe.model').read_bytes())
    algorithm = sentencepiece_model_pb2.TrainerSpec.ModelType.Name(proto.trainer_spec.model_type)
    assert tokenizer.vocab_size() == 500, tokenizer.vocab_size()
    upstream_tokens = {}
    for line in (assets / 'tokens.txt').read_text().splitlines():
        piece, token_id = line.rsplit(maxsplit=1)
        upstream_tokens[int(token_id)] = piece
    pieces = [tokenizer.id_to_piece(index) for index in range(tokenizer.vocab_size())]
    assert all(upstream_tokens[index] == piece for index, piece in enumerate(pieces))
    examples = []
    for text in ['THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG', "DON'T STOP READING", 'BRICKS ARE AN ALTERNATIVE']:
        token_ids = tokenizer.encode(text)
        assert tokenizer.unk_id() not in token_ids, text
        assert tokenizer.decode(token_ids) == text, text
        examples.append(dict(text=text, pieces=tokenizer.encode(text, out_type=str), ids=token_ids))

    metadata = dict(
        repository=REPOSITORY, revision=REVISION, files=sources,
        algorithm=algorithm, sentencepiece_vocab_size=tokenizer.vocab_size(),
        sentencepiece_version=spm.__version__,
        special_pieces=[dict(id=index, piece=piece.piece, type=piece.type)
                        for index, piece in enumerate(proto.pieces) if piece.type != 1],
        upstream_extra_symbols={str(index): piece for index, piece in upstream_tokens.items() if index >= 500},
        trainer_spec=str(proto.trainer_spec), normalizer_spec_name=proto.normalizer_spec.name,
        text_policy='Use uppercase LibriSpeech-style text; preserve apostrophes. Examples checked for exact roundtrip.',
        recipe_url='https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh',
        corpus_provenance='Published with the Icefall LibriSpeech model. Current upstream recipe trains BPE on train-clean-100, train-clean-360 and train-other-500 only. Original transcript hashes are not supplied by the model publisher; exact corpus membership is not independently verified.',
        license_status='No explicit model asset license found in the publication metadata; do not infer it from the Icefall code license.',
        examples=examples, evaluation_uploads=False,
    )
    (assets / 'tokens.json').write_text(json.dumps(pieces, indent=2, ensure_ascii=True) + '\n')
    (HERE / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + '\n')
    print(f'Public lang_bpe_500 actual algorithm: {algorithm}', flush=True)


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('\u2019', "'")
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)*", text))


def train_bpe():
    source = HERE.parent / 'multihead_libri960/data/train-train.jsonl'
    source_metadata = json.loads((source.parent / 'metadata.json').read_text())
    expected = source_metadata['splits']['train-train']['subsets']
    assert set(expected) == {'train-clean-100', 'train-clean-360', 'train-other-500'}
    assets = HERE / 'bpe500'
    assets.mkdir(exist_ok=True)
    model_path = assets / 'bpe.model'
    if model_path.exists():
        raise FileExistsError(f'Refusing to overwrite trained model: {model_path}')
    transcript = assets / 'train.txt'
    source_hash = hashlib.sha256()
    counts = Counter()
    seen = set()
    with source.open('rb') as records, transcript.open('w', encoding='utf-8') as output:
        for line in records:
            source_hash.update(line)
            row = json.loads(line)
            assert row['subset'] in expected, row['subset']
            assert row['utt'] not in seen, row['utt']
            seen.add(row['utt'])
            text = normalize_text(row['text'])
            assert text, row['utt']
            counts[row['subset']] += 1
            output.write(text + '\n')
    assert dict(counts) == {subset: details['utterances'] for subset, details in expected.items()}
    options = dict(model_type='bpe', vocab_size=500, character_coverage=1.0,
                   normalization_rule_name='identity', num_threads=4,
                   input_sentence_size=0, shuffle_input_sentence=False,
                   max_sentence_length=65536, hard_vocab_limit=True,
                   pad_id=0, pad_piece='<blank>', unk_id=1, bos_id=2, eos_id=3)
    spm.SentencePieceTrainer.train(input=str(transcript), model_prefix=str(assets / 'bpe'), **options)
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    assert proto.trainer_spec.model_type == sentencepiece_model_pb2.TrainerSpec.BPE
    assert tokenizer.vocab_size() == 500
    assert (tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()) == (0, 1, 2, 3)
    checked = total_tokens = max_tokens = 0
    with transcript.open() as sentences:
        for line in sentences:
            text = line.rstrip('\n')
            token_ids = tokenizer.encode(text)
            assert token_ids and all(token_id >= 4 for token_id in token_ids), text
            assert tokenizer.decode(token_ids) == text, text
            checked += 1
            total_tokens += len(token_ids)
            max_tokens = max(max_tokens, len(token_ids))
    assert checked == sum(counts.values())
    pieces = [tokenizer.id_to_piece(index) for index in range(500)]
    (assets / 'tokens.json').write_text(json.dumps(pieces, indent=2, ensure_ascii=True) + '\n')
    (assets / 'tokens.txt').write_text(''.join(f'{piece} {index}\n' for index, piece in enumerate(pieces)))
    transcript_hash = hashlib.sha256(transcript.read_bytes()).hexdigest()
    metadata = dict(
        algorithm='BPE', vocab_size=500, ordinary_pieces=496,
        special_ids=dict(blank_pad=0, unk=1, bos=2, eos=3),
        sentencepiece_version=spm.__version__, training_options=options,
        source_manifest=str(source), source_manifest_sha256=source_hash.hexdigest(),
        source_metadata_sha256=hashlib.sha256((source.parent / 'metadata.json').read_bytes()).hexdigest(),
        prepare_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_text_sha256=transcript_hash, subsets=dict(counts),
        corpus_policy='All entries in existing LibriSpeech 960h training manifest; no dev/test text.',
        normalization="NFKC, lowercase, normalize curly apostrophes; retain ASCII letter words and internal apostrophes; other punctuation separates words.",
        validation=dict(utterances=checked, unknown_tokens=0, roundtrip_failures=0,
                        mean_tokens=total_tokens / checked, max_tokens=max_tokens),
        files_sha256={filename: hashlib.sha256((assets / filename).read_bytes()).hexdigest()
                      for filename in ['bpe.model', 'bpe.vocab', 'tokens.json', 'tokens.txt']},
        examples=[dict(text=text, pieces=tokenizer.encode(text, out_type=str), ids=tokenizer.encode(text))
                  for text in ["don't stop reading", 'bricks are an alternative']],
        evaluation_uploads=False,
    )
    (assets / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


def main():
    inspect_public()
    train_bpe()


if __name__ == '__main__':
    main()