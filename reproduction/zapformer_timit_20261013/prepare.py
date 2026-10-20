"""Cache all TIMIT audio as the existing 80-bin/10ms Kaldi Fbank frontend."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.compliance import kaldi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def extract(row):
    torch.set_num_threads(1)
    audio, sr = sf.read(row['wav'], dtype='float32')
    if sr != 16000 or audio.ndim != 1:
        raise ValueError(f'Expected 16kHz mono: {row["wav"]}')
    feature = kaldi.fbank(torch.from_numpy(audio).unsqueeze(0) * 32768,
                          sample_frequency=16000, num_mel_bins=80,
                          frame_shift=10., frame_length=25., snip_edges=False,
                          dither=0., energy_floor=0.).numpy()
    return feature, len(audio) / sr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'data')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    sources = {s: ROOT / 'reproduction/multihead_timit/data' / f'{name}.jsonl'
               for s, name in [('train', 'train-train'), ('dev', 'train-dev'), ('test', 'test')]}
    vocab_path = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor/vocab.json'
    vocab = json.loads(vocab_path.read_text())
    contract = dict(source_sha256={s: hashlib.sha256(path.read_bytes()).hexdigest() for s, path in sources.items()},
                    vocab=vocab, frontend='kaldi.fbank, waveform*32768,80bins,25ms,10ms,snip_edges=False,dither=0,energy_floor=0',
                    preparation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    marker = args.output / 'READY.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['contract'] != contract:
            raise ValueError('Existing cache contract differs; choose another output')
        for split, record in saved['splits'].items():
            path = args.output / f'{split}.jsonl'
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['rows_sha256']:
                raise ValueError('Cached rows changed')
            if (args.output / f'{split}.f32').stat().st_size != record['feature_bytes']:
                raise ValueError('Cached features truncated')
        print(f'Validated existing cache: {args.output}')
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Incomplete/nonempty cache; use a new output instead of overwriting')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(contract=contract, splits={})
    seen = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split, source in sources.items():
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            expected = {'train': 4620, 'dev': 400, 'test': 192}[split]
            assert len(rows) == expected
            features_path = args.output / f'{split}.f32'
            output_rows = []
            offset = 0
            with features_path.open('wb') as stream:
                for row, (features, duration) in zip(rows, pool.map(extract, rows, chunksize=8)):
                    assert row['utt'] not in seen
                    seen.add(row['utt'])
                    references = row['references']['ottc_arpabet']
                    tokens = [vocab[r[0]] for r in references]
                    assert tokens and not set(tokens) & {vocab['<pad>'], vocab['<unk>']}
                    features.astype(np.float32).tofile(stream)
                    output_rows.append(dict(id=row['utt'], wav=row['wav'], duration=duration,
                                            frames=len(features), feature_offset=offset,
                                            feature_file=str(features_path.resolve()), tokens=tokens,
                                            references=references))
                    offset += features.size
            path = args.output / f'{split}.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in output_rows))
            metadata['splits'][split] = dict(utterances=len(rows), hours=sum(r['duration'] for r in output_rows)/3600,
                                            rows_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                            feature_bytes=features_path.stat().st_size,
                                            feature_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest())
            print(split, metadata['splits'][split], flush=True)
    marker.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
