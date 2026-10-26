"""Reuse existing Libri100 utterance splits with the verified local BPE500."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.timit_bpe500_20261020.train import Tokenizer
from reproduction.librispeech_bpe500_20261020.prepare import normalize_text
from reproduction.powsm_bpe_timit_20261019.prepare import sha256, write_json

SOURCE = ROOT / 'reproduction/multihead_libri100/data'
ASSETS = ROOT / 'reproduction/librispeech_bpe500_20261020/bpe500'
DATA = HERE / 'data'
SPLITS = ('train', 'dev-clean', 'dev-other', 'test-clean', 'test-other')


def check(directory=DATA):
    metadata = json.loads((directory / 'metadata.json').read_text())
    tokenizer = Tokenizer(directory / 'tokenizer')
    assert metadata['algorithm'] == 'BPE' and metadata['vocab_size'] == 500
    assert sha256(directory / 'tokenizer/bpe.model') == metadata['tokenizer_sha256']
    assert sha256(directory / 'tokenizer/tokens.json') == metadata['tokens_sha256']
    seen, speakers = set(), {}
    for split in SPLITS:
        path = directory / (split + '.jsonl')
        assert sha256(path) == metadata['manifests_sha256'][split]
        count = 0
        speakers[split] = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row['utt'] not in seen
                seen.add(row['utt'])
                speakers[split].add(row['speaker'])
                assert row['labels'] and all(4 <= value < 500 for value in row['labels'])
                assert tokenizer.decode(row['labels']) == row['text'] == normalize_text(row['text'])
                assert row['subset'] == ('train-clean-100' if split == 'train' else split)
                count += 1
        assert count == metadata['statistics'][split]['utterances']
    assert metadata['statistics']['train']['utterances'] == 28538
    for split in SPLITS[1:]:
        assert not speakers['train'] & speakers[split], split
    print(json.dumps(dict(validation='PASS: BPE500, distinct utterances/speakers, roundtrip and source hashes',
                          statistics=metadata['statistics']), indent=2), flush=True)
    return metadata


def prepare():
    if DATA.exists():
        return check()
    parent_metadata = json.loads((SOURCE / 'metadata.json').read_text())
    tokenizer = Tokenizer(ASSETS)
    staging = Path(tempfile.mkdtemp(prefix='.prepare-', dir=HERE))
    streams = {}
    try:
        streams = {split: (staging / (split + '.jsonl')).open('w') for split in SPLITS}
        statistics = {split: dict(utterances=0, seconds=0., max_seconds=0., max_tokens=0) for split in SPLITS}
        sources = {}
        for name, allowed in [('train-train', {'train-clean-100'}),
                              ('train-dev', {'dev-clean', 'dev-other'}), ('test', {'test-clean', 'test-other'})]:
            path = SOURCE / (name + '.jsonl')
            count = 0
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    assert row['subset'] in allowed
                    split = 'train' if row['subset'] == 'train-clean-100' else row['subset']
                    text = normalize_text(row['text'])
                    labels = tokenizer.encode(text)
                    assert labels and all(4 <= value < 500 for value in labels), row['utt']
                    assert tokenizer.decode(labels) == text, row['utt']
                    assert Path(row['wav']).is_file(), row['wav']
                    clean = dict(utt=row['utt'], wav=row['wav'], speaker=str(row['speaker']), subset=row['subset'],
                        duration=row['duration'], original_text=row['text'], text=text, labels=labels)
                    streams[split].write(json.dumps(clean, ensure_ascii=True) + '\n')
                    stats = statistics[split]
                    stats['utterances'] += 1
                    stats['seconds'] += row['duration']
                    stats['max_seconds'] = max(stats['max_seconds'], row['duration'])
                    stats['max_tokens'] = max(stats['max_tokens'], len(labels))
                    count += 1
            assert count == parent_metadata['splits'][name]['kept_utterances'], (name, count)
            sources[name] = dict(path=str(path), sha256=sha256(path), utterances=count,
                excluded_manifest=str(SOURCE / (name + '_excluded.jsonl')),
                excluded_manifest_sha256=sha256(SOURCE / (name + '_excluded.jsonl')))
        for stream in streams.values():
            stream.close()
        assets = staging / 'tokenizer'
        assets.mkdir()
        for filename in ['bpe.model', 'tokens.json']:
            shutil.copy2(ASSETS / filename, assets / filename)
        metadata = dict(dataset='LibriSpeech train-clean-100', algorithm='BPE', vocab_size=500,
            tokenizer_sha256=sha256(assets / 'bpe.model'), tokens_sha256=sha256(assets / 'tokens.json'),
            tokenizer_training_metadata=json.loads((ASSETS / 'metadata.json').read_text()),
            source_metadata_sha256=sha256(SOURCE / 'metadata.json'), sources=sources, statistics=statistics,
            manifests_sha256={split: sha256(staging / (split + '.jsonl')) for split in SPLITS},
            normalization='Existing BPE500 normalization: lowercase NFKC, retain internal apostrophes.',
            eligibility='Reuse existing multihead Libri100 manifest eligibility (28538 training utterances); not a claim of all 28539 official files.',
            supervision='Text BPE only. Existing automatic phone alignments are not training targets or human segmentation ground truth.',
            prepare_sha256=sha256(Path(__file__)), evaluation_uploads=False)
        write_json(staging / 'metadata.json', metadata)
        check(staging)
        write_json(staging / 'DATA_READY.json', dict(metadata_sha256=sha256(staging / 'metadata.json'), evaluation_uploads=False))
        staging.rename(DATA)
        return metadata
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check() if args.check else prepare()