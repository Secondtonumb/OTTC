"""Bounded TIMIT BPE500 probes reusing the frozen original OTTC recipes."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.powsm_bpe_timit_20261019 import train as base
from reproduction.powsm_bpe_timit_20261019.gradient_check_fix import train as checked
from reproduction.librispeech_bpe500_20261020.prepare import normalize_text
from reproduction.wandb_policy import init_evaluation
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

PARENT_SOURCES = base.source_hashes()
ASSETS = ROOT / 'reproduction/librispeech_bpe500_20261020/bpe500'


class Tokenizer:
    def __init__(self, directory):
        self.sp = spm.SentencePieceProcessor(model_file=str(directory / 'bpe.model'))
        self.tokens = json.loads((directory / 'tokens.json').read_text())
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString((directory / 'bpe.model').read_bytes())
        assert proto.trainer_spec.model_type == sentencepiece_model_pb2.TrainerSpec.BPE
        assert len(self.tokens) == self.sp.vocab_size() == 500
        assert self.tokens == [self.sp.id_to_piece(index) for index in range(500)]
        assert self.tokens[:4] == ['<blank>', '<unk>', '<s>', '</s>']
        self.blank_id, self.unk_id = 0, 1

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode([int(index) for index in ids if int(index) > 0])


def prepare():
    destination = HERE / 'data'
    assert not destination.exists(), 'Data already prepared; use --check to validate it'
    tokenizer = Tokenizer(ASSETS)
    statistics, sources, utterances, speakers = {}, {}, {}, {}
    for split, filename, expected in [('train', 'train-train', 4620), ('dev', 'train-dev', 400), ('test', 'test', 192)]:
        source = ROOT / 'reproduction/multihead_timit/data' / (filename + '.jsonl')
        records = [json.loads(line) for line in source.open()]
        assert len(records) == expected
        utterances[split], speakers[split] = set(), set()
        rows, lengths = [], []
        for row in records:
            text = normalize_text(row['text'])
            labels = tokenizer.encode(text)
            assert labels and all(4 <= token_id < 500 for token_id in labels)
            assert tokenizer.decode(labels) == text
            assert row['utt'] not in utterances[split]
            assert Path(row['wav']).is_file(), row['wav']
            utterances[split].add(row['utt'])
            speakers[split].add(row['utt'].split('_')[0])
            lengths.append(len(labels))
            rows.append(dict(utt=row['utt'], wav=row['wav'], duration=row['duration'],
                             original_text=row['text'], text=text, labels=labels))
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / (split + '.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in rows))
        temporary.replace(destination / (split + '.jsonl'))
        sources[split] = dict(path=str(source), sha256=base.sha256(source))
        statistics[split] = dict(utterances=len(rows), unknown_tokens=0, roundtrip_failures=0,
                                mean_tokens=sum(lengths) / len(lengths), max_tokens=max(lengths))
    for first, second in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        assert not utterances[first] & utterances[second]
        assert not speakers[first] & speakers[second]
    assets = destination / 'tokenizer'
    assets.mkdir()
    for filename in ['bpe.model', 'tokens.json']:
        shutil.copy2(ASSETS / filename, assets / filename)
    metadata = dict(algorithm='BPE', output_vocab_size=500,
        tokenizer_sha256=base.sha256(assets / 'bpe.model'), tokens_sha256=base.sha256(assets / 'tokens.json'),
        tokenizer_training_metadata=json.loads((ASSETS / 'metadata.json').read_text()),
        normalization='NFKC lowercase, retain internal apostrophes; punctuation separates words',
        adaptation='New BPE500 inventory, preserving apostrophes unlike the POWSM 40k recipe. Original OTTC JSD/CE losses unchanged.',
        source_manifests=sources, statistics=statistics,
        manifests_sha256={split: base.sha256(destination / (split + '.jsonl')) for split in statistics},
        evaluation_uploads=False)
    base.write_json(destination / 'metadata.json', metadata)
    print(json.dumps(statistics, indent=2), flush=True)


def source_hashes():
    sources = dict(PARENT_SOURCES)
    paths = [*HERE.glob('*.py'), *HERE.glob('*.pbs'), Path(checked.__file__),
             ROOT / 'reproduction/librispeech_bpe500_20261020/prepare.py']
    sources.update({str(path.relative_to(ROOT)): base.sha256(path) for path in paths})
    return sources


def start_tracking(output, config, smoke):
    base.write_json(output / 'TEST_PROTOCOL.json', dict(
        purpose='GPU smoke' if smoke else 'bounded 200-step early-learning probe, not a final benchmark',
        evaluation_uploads=False, reserved_ids=[0, 1, 2, 3],
        adaptation='BPE500 with internal apostrophes preserved; no change to original OTTC cost',
        decoding='Unconstrained greedy frame argmax, collapse repeats, remove blank, SentencePiece decode',
        subsets=dict(train_examples=8, dev_examples=20)))
    return init_evaluation(project='ottc-timit-reproduction', name='timit_bpe500_' + config['recipe'],
                           job_type='smoke' if smoke else 'probe', config=config, dir=str(output))


class ProbeTrainer(checked.CheckedTrainer):
    def train(self, *args, **kwargs):
        output = Path(self.args.output_dir)
        if output.parent.name == 'smoke':
            return super().train(*args, **kwargs)
        tokenizer = Tokenizer(HERE / 'data/tokenizer')
        rows = {split: [json.loads(line) for line in (HERE / 'data' / (split + '.jsonl')).open()][:count]
                for split, count in [('train', 8), ('dev', 20)]}
        summary = {}
        if not args and not kwargs.get('resume_from_checkpoint'):
            summary['before'] = {split: base.evaluate(self.model, items, self.tokenizer, tokenizer,
                output / 'decode/before' / (split + '.jsonl'), 2) for split, items in rows.items()}
        result = super().train(*args, **kwargs)
        summary['after'] = {split: base.evaluate(self.model, items, self.tokenizer, tokenizer,
            output / 'decode/after' / (split + '.jsonl'), 2) for split, items in rows.items()}
        summary.update(step=self.state.global_step, epoch=self.state.epoch, evaluation_uploads=False,
                       protocol='First 8 train and first 20 dev utterances; diagnostics only, not full-set WER')
        base.write_json(output / 'PROBE_RESULTS.json', summary)
        return result


def main():
    if sys.argv[1:] == ['--prepare']:
        prepare()
        return
    if sys.argv[1:] == ['--submit']:
        dispatch = HERE / 'DISPATCH.json'
        assert not dispatch.exists(), 'Already submitted; inspect saved jobs before resubmission'
        jobs = HERE / 'jobs'
        jobs.mkdir(exist_ok=True)
        record = dict(source_sha256=source_hashes(),
                      metadata_sha256=base.sha256(HERE / 'data/metadata.json'),
                      evaluation_uploads=False, jobs=[])
        base.write_json(dispatch, record)
        for cost in ['jsd', 'ce']:
            command = ['/opt/pbs/bin/qsub', '-N', 'bpe500_' + cost, '-v', 'BPE_COST=' + cost,
                       '-o', str(jobs / (cost + '.log')), str(HERE / 'run.pbs')]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            record['jobs'].append(dict(cost=cost, job=result.stdout.strip(), command=command,
                                       stderr=result.stderr))
            base.write_json(dispatch, record)
        print(json.dumps(record['jobs'], indent=2), flush=True)
        return
    dispatch = HERE / 'DISPATCH.json'
    if dispatch.exists():
        contract = json.loads(dispatch.read_text())
        assert contract['source_sha256'] == source_hashes(), 'Submitted source contract changed'
        assert contract['metadata_sha256'] == base.sha256(HERE / 'data/metadata.json')
    base.HERE = HERE
    base.Tokenizer = Tokenizer
    base.normalize_text = normalize_text
    base.CheckedTrainer = ProbeTrainer
    base.check = checked.check
    base.source_hashes = source_hashes
    base.start_tracking = start_tracking
    assert '--head' not in sys.argv, 'This probe uses plain OTTC only'
    if '--check' not in sys.argv:
        assert '--smoke' in sys.argv or ('--stop-after-steps' in sys.argv and
            sys.argv[sys.argv.index('--stop-after-steps') + 1] == '200'), 'Explicit 200-step probe required'
    metadata = json.loads((HERE / 'data/metadata.json').read_text())
    for split, expected in metadata['manifests_sha256'].items():
        assert base.sha256(HERE / 'data' / (split + '.jsonl')) == expected
    tokenizer = Tokenizer(HERE / 'data/tokenizer')
    assert base.sha256(HERE / 'data/tokenizer/bpe.model') == metadata['tokenizer_sha256']
    assert base.sha256(HERE / 'data/tokenizer/tokens.json') == metadata['tokens_sha256']
    for text in ["don't stop reading", 'she she', 'bricks are an alternative']:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    base.main()


if __name__ == '__main__':
    main()