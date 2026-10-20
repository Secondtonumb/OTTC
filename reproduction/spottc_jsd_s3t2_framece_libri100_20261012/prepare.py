"""Reuse historical Libri100 phone times, adapting labels to the frozen MFA inventory."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = Path('/home/m64000/work/IF-MDD')
SOURCE = OLD/'data/context_phone_shared_librispeech_l2arctic_pruned_budgeted/librispeech/triphone_pruned_budget256/train-train.json'
BASE = Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer')
TOKENS = Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/tokens.txt')
sys.path.insert(0, str(BASE))
from evaluation_phone_mapping import BASE as ARPA_TO_IPA


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def annotations():
    HERE.mkdir(parents=True, exist_ok=True)
    codec_source = OLD/'utils/context_phone_codec.py'
    (HERE/'historical_codec.py').write_text(codec_source.read_text())
    spec = importlib.util.spec_from_file_location('historical_codec', HERE/'historical_codec.py')
    codec = importlib.util.module_from_spec(spec); spec.loader.exec_module(codec)
    src = OLD/'utils/DataPrepIO.py'
    tree = ast.parse(src.read_text())
    timing = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_token_times_from_training_alignment')
    timing.decorator_list = []
    frozen = ast.unparse(timing) + '\n'
    (HERE/'historical_times.py').write_text(frozen)
    ns = {'center_phone': codec.center_phone}; exec(frozen, ns)
    time_function = ns[timing.name]
    inventory = {p: int(i) for p, i in (line.rsplit(' ', 1) for line in TOKENS.read_text().splitlines())}
    phone_map = {**ARPA_TO_IPA, 'sil': 'sil', 'sp': 'sil', 'spn': 'spn', '<blank>': 'sil'}
    payload = json.loads(SOURCE.read_text()); assert len(payload) == 28538
    rows = []; usage = Counter(); samples = Counter()
    for key, row in payload.items():
        tokens = row['perceived_train_target'].split()
        starts, ends = time_function(tokens, row['phonemes'], row['duration'], 'triphone')
        labels = [codec.center_phone(t, 'triphone') for t in tokens]
        assert len(labels) == len(starts) == len(ends)
        spans = []
        for label, start, end in zip(labels, starts, ends):
            if end <= start:
                continue
            if label not in phone_map:
                raise ValueError('Unsupported training phone: ' + label)
            ipa = phone_map[label]; pid = inventory[ipa]
            spans.append([pid, start, end]); usage[label] += 1
        audio = Path(row['wav'])
        if not audio.is_file():
            raise FileNotFoundError(audio)
        assert row['subset'] == 'train-clean-100' and not str(audio).lower().count('timit')
        uid = audio.stem
        assert not samples[uid]; samples[uid] += 1
        rows.append(dict(id=uid, wav=str(audio), duration=row['duration'],
                         speaker=row['spk_id'], intervals=spans))
    (HERE/'annotations.json').write_text(json.dumps(rows, ensure_ascii=False))
    originals = [SOURCE, src, codec_source, TOKENS, BASE/'evaluation_phone_mapping.py',
                 OLD/'hparams_librispeech100_10ms_budget256_ctxctc_film_ce_3ep/s3_t2_star_star/phnmonossl_ctc_alignment_ce.yaml']
    (HERE/'timeline_provenance.json').write_text(json.dumps(dict(
        source_hashes={str(p): sha(p) for p in originals}, utterances=len(rows),
        hours=sum(r['duration'] for r in rows)/3600, source_label_usage=dict(usage),
        arpa_to_ipa={k: dict(phone=v, id=inventory[v]) for k, v in phone_map.items()},
        time_policy='Exact historical _token_times_from_training_alignment; gaps from original silence tokens; no new aligner',
        frame_policy='float32 times; floor(start/duration*frames), ceil(end/duration*frames); later spans overwrite; unlabelled=-100',
        inventory_adaptation='ARPAbet centers -> frozen MFA lexical IPA representatives; model keeps all 105 phone IDs / 313 state outputs',
        gradient_training_dataset='LibriSpeech train-clean-100 only', timit_gradient_training=False
    ), ensure_ascii=False, indent=2)+'\n')
    # Keep only the official 400 development utterances, plus the official 192 test.
    eval_dir = HERE/'eval_data'; eval_dir.mkdir(exist_ok=True)
    old_eval = HERE.parent/'english_suite/mfa_eval_data'
    split_path = Path('/work/gm64/m64000/fa-bench/datasets/languages/en/timit/split/dev.list')
    keys = set()
    for line in split_path.read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        speaker, uid = line.split(); keys.add('timit:'+speaker.lower()+'_'+uid.split('_')[-1].lower())
    development = [json.loads(line) for line in (old_eval/'development.jsonl').open()]
    selected = [r for r in development if r['id'] in keys]
    assert len(selected) == len(keys) == 400
    assert all(r['gradient_training_allowed'] is False for r in selected)
    (eval_dir/'development.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    (eval_dir/'test.jsonl').write_text((old_eval/'test.jsonl').read_text())
    assert len((eval_dir/'test.jsonl').read_text().splitlines()) == 192
    print('ANNOTATIONS_READY', len(rows), flush=True)


def features(workers):
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.compliance import kaldi
    torch.set_num_threads(1)
    rows = json.loads((HERE/'annotations.json').read_text())
    out = HERE/'data'; out.mkdir(exist_ok=True)
    def shard(index):
        manifest = out/f'shard-{index}.jsonl'
        blob = out/f'shard-{index}.f32'
        if manifest.exists() and blob.exists():
            cached = [json.loads(line) for line in manifest.open()]
            assert len(cached) == len(rows[index::workers])
            assert blob.stat().st_size == sum(r['frames']*80*4 for r in cached)
            return cached
        result = []; offset = 0
        with blob.with_suffix('.partial').open('wb') as stream:
            for row in rows[index::workers]:
                audio, rate = sf.read(row['wav'], dtype='float32')
                assert rate == 16000 and audio.ndim == 1
                assert abs(len(audio)/rate-row['duration']) <= 1/rate+1e-6
                feat = kaldi.fbank(torch.from_numpy(audio).unsqueeze(0)*32768,
                    sample_frequency=16000, num_mel_bins=80, frame_shift=10.,
                    frame_length=25., snip_edges=False, dither=0., energy_floor=0.)
                assert bool(torch.isfinite(feat).all())
                feat.numpy().astype('<f4').tofile(stream)
                result.append(dict(row, frames=len(feat), feature_file=str(blob), feature_offset=offset))
                offset += feat.numel()
                if len(result)%500 == 0: print('FEATURES', index, len(result), flush=True)
        blob.with_suffix('.partial').replace(blob)
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in result))
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(shard, range(workers)))
    result = sorted([r for shard_rows in shards for r in shard_rows], key=lambda r:r['id'])
    assert len(result) == len({r['id'] for r in result}) == 28538
    manifest = out/'training.jsonl'; manifest.write_text(''.join(json.dumps(r)+'\n' for r in result))
    hashes = {str(p):sha(p) for p in sorted(out.glob('shard-*.f32'))}
    (out/'PREPARE_COMPLETE.json').write_text(json.dumps(dict(utterances=len(result),
        training_manifest_sha256=sha(manifest), features_sha256=hashes,
        annotations_sha256=sha(HERE/'annotations.json'), source_provenance_sha256=sha(HERE/'timeline_provenance.json'),
        feature_config=dict(num_mel_bins=80, frame_shift_ms=10, frame_length_ms=25,
                            snip_edges=False, dither=0, energy_floor=0, waveform_scale=32768)), indent=2)+'\n')
    print('PREPARE_COMPLETE', len(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--features', action='store_true')
    parser.add_argument('--workers', type=int, default=8); args = parser.parse_args()
    if args.features: features(args.workers)
    else: annotations()
