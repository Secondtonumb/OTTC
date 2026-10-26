"""Export real audio-only posteriors from four completed gold-frame models."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/timit-frame-posterior-mpl')
import numpy as np
import soundfile as sf
import torch
from reproduction.timit_ssl_native10_20261019 import common as base
from reproduction.timit_frame_ce_20261026 import common as ce
from reproduction.timit_frame_divergences_20261026 import common as div

HERE = Path(__file__).resolve().parent
ROOT = base.ROOT
METHODS = [('ce', 'Frame-wise CE'), ('jsd_standard', 'Frame-wise JSD'),
           ('ag', 'Frame-wise AG'), ('jdiv', 'Frame-wise J-DIV')]


def render_and_finish(output):
    from reproduction.timit_frame_posterior_20261026.plot import draw
    manifest = json.loads((output / 'manifest.json').read_text())
    for member in manifest['methods']:
        cost = member['cost']
        for name, digest in member['files'].items():
            assert base.sha(output / name) == digest
        with np.load(output / f'{cost}.npz') as arrays:
            raw = json.loads((output / f'{cost}.json').read_text())
            assert np.array_equal(arrays['posterior'], np.asarray(raw['posterior'], dtype=np.float32))
            assert np.array_equal(arrays['logits'], np.asarray(raw['logits'], dtype=np.float32))
            with (output / f'{cost}.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == arrays['posterior'].size
            for r in rows:
                t, p = int(r['frame']), int(r['phone_id'])
                assert float(r['posterior']) == float(arrays['posterior'][t, p])
                assert float(r['logit']) == float(arrays['logits'][t, p])
    draw(output)
    base.write(output / 'COMPLETE.json', dict(real_audio=True, posterior_verified=True,
        full_vocabulary=42, npz_json_csv_identical=True, methods=[k for k, _ in METHODS],
        files={p.name: base.sha(p) for p in output.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
    print('COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43, choices=[42, 43, 44])
    parser.add_argument('--utt', default='FADG0_SI1279')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    output = HERE / f'seed{args.seed}' / args.utt
    if args.render_only:
        render_and_finish(output)
        return
    output.mkdir(parents=True, exist_ok=True)
    ce_root = ROOT / 'reproduction/timit_frame_ce_20261026/models'
    div_root = ROOT / 'reproduction/timit_frame_divergences_20261026/models'
    control = json.loads((ce_root / f'frame_ce__seed{args.seed}/run_config.json').read_text())
    _, rows, vocab, processor, audit = base.load_data(control)
    assert control['data'] == audit
    row = next(r for r in rows['dev'] if r['utt'] == args.utt)
    original = next(json.loads(line) for line in (ROOT / 'reproduction/multihead_timit/data/train-dev.jsonl').open()
                    if json.loads(line)['utt'] == args.utt)
    labels = [p for p, _ in sorted(vocab.items(), key=lambda item: item[1])]
    # Exact same audio preprocessing as the matched evaluator, without any target input.
    audio, sr = sf.read(row['wav'], dtype='float32')
    assert sr == 16000 and audio.ndim == 1
    features = [dict(input_values=processor(audio, sampling_rate=sr).input_values[0])]
    batch = processor.pad(features, return_tensors='pt', return_attention_mask=True)
    assert set(batch) == {'input_values', 'attention_mask'}
    shutil.copyfile(row['wav'], output / 'audio.wav')
    base.write(output / 'vocab.json', vocab)
    base.write(output / 'reference.json', dict(utt=row['utt'], text=original['text'], split='dev',
        duration=row['duration'], references=original['references'],
        note='Human-PHN mapped tiers copied unchanged; never supplied to inference.'))
    exports = []
    for cost, title in METHODS:
        begin = time.monotonic()
        source = (ce_root / f'frame_ce__seed{args.seed}') if cost == 'ce' else (div_root / f'frame_{cost}__seed{args.seed}')
        config = json.loads((source / 'run_config.json').read_text())
        assert (source / 'TRAIN_COMPLETE.json').exists() and (source / 'BENCHMARK_COMPLETE.json').exists()
        for key in ('backbone', 'head', 'frame_ms', 'native_conv_stride', 'forward_type', 'seed',
                    'global_batch', 'epochs', 'learning_rate', 'warmup_steps', 'weight_decay', 'data', 'frame_supervision'):
            assert config[key] == control[key], key
        best = json.loads((source / 'BEST_DEV.json').read_text())
        checkpoint = (source / 'best_dev.pt').resolve()
        assert checkpoint == Path(best['path']).resolve()
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == best['step'] and saved['recipe'] == config['name']
        assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
        model, topo = (ce if cost == 'ce' else div).build_model(config, vocab, 73)
        assert topo is None
        model.load_state_dict(saved['model'], strict=True)
        model.to(device='cpu', dtype=torch.float32).eval()
        with torch.inference_mode():
            out = model(**batch)
        length = int(out.hidden_states[1][0].sum())
        logits = out.logits[0, :length].float().numpy()
        posterior = torch.from_numpy(logits).softmax(-1).numpy()
        assert out.hidden_states[0] is None, 'Frame models have no active OT mass output'
        frame_time = np.arange(length, dtype=np.float64) * .01
        receptive_field, jump = 1, 1
        for kernel, stride in zip(model.wav2vec2.config.conv_kernel, config['native_conv_stride']):
            receptive_field += (kernel-1) * jump
            jump *= stride
        centers = (np.arange(length)*jump+(receptive_field-1)/2)/sr
        assert receptive_field == 400 and jump == 160
        assert posterior.shape == (length, 42) and np.isfinite(posterior).all()
        assert np.allclose(posterior.sum(-1), 1., atol=2e-6)
        argmax = posterior.argmax(-1)
        recorded = next(json.loads(line) for line in (source / 'benchmark_best_dev_dev.predictions.jsonl').open()
                        if json.loads(line)['utt'] == row['utt'])
        old_ids = np.asarray(recorded['text_free']['frame_token_ids'])
        assert np.array_equal(old_ids, argmax), (cost, float((old_ids == argmax).mean()))
        np.savez_compressed(output / f'{cost}.npz', posterior=posterior, logits=logits,
            frame_time_seconds=frame_time, receptive_center_seconds=centers,
            argmax_ids=argmax, labels=np.asarray(labels))
        base.write(output / f'{cost}.json', dict(method=title, recipe=config['name'], utterance=row['utt'],
            labels=labels, frame_time_seconds=frame_time.tolist(), receptive_center_seconds=centers.tolist(),
            posterior=posterior.tolist(), logits=logits.tolist(), argmax_ids=argmax.tolist()))
        with (output / f'{cost}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame', 'time_seconds', 'receptive_center_seconds', 'phone_id', 'phone', 'posterior', 'logit'])
            for t in range(length):
                for p in range(42):
                    writer.writerow([t, float(frame_time[t]), float(centers[t]), p, labels[p],
                                     float(posterior[t, p]), float(logits[t, p])])
        exports.append(dict(cost=cost, title=title, recipe=config['name'], seed=args.seed, shape=list(posterior.shape),
            checkpoint=str(checkpoint), checkpoint_sha256=base.sha(checkpoint), epoch=saved['epoch'], step=saved['step'],
            config_sha256=base.sha(source / 'run_config.json'), model_class=type(model).__name__,
            model_source_sha256=base.sha(ROOT / ('reproduction/timit_frame_ce_20261026/model.py' if cost == 'ce' else
                'reproduction/timit_frame_divergences_20261026/model.py')),
            selection=best['selection'], objective=config['objective'], saved_fp32=True,
            historical_argmax_agreement=1., seconds=time.monotonic()-begin,
            files={f'{cost}.{ext}': base.sha(output / f'{cost}.{ext}') for ext in ('npz', 'json', 'csv')}))
        print('EXPORTED', cost, 'epoch', saved['epoch'], 'shape', posterior.shape, 'argmax matches GPU100%', flush=True)
        del model, saved, out
        gc.collect()
    base.write(output / 'manifest.json', dict(utterance=row['utt'], seed=args.seed, split='dev', text=original['text'],
        selection='same fixed first dev utterance as the earlier OTTC posterior export',
        source_audio=row['wav'], audio_sha256=base.sha(output / 'audio.wav'), duration=row['duration'], data=audit,
        labels=labels, methods=exports,
        inference=dict(device='cpu', dtype='float32', autocast=False, eval=True, labels_supplied=False,
            frame_targets_supplied=False, reference_used_for_inference=False, optimizer_updated=False,
            posterior='softmax(lm_head(dropout(encoder(audio))), dim=all42classes)',
            ot_frame_mass='absent; duration head frozen and unused in frame-wise models',
            receptive_field_samples=400, receptive_center_offset_seconds=12.46875/1000,
            plot_clock='CNN receptive-field centers; raw historical frame_index*10ms also retained separately',
            no_resampling_no_smoothing_no_temperature=True),
        provenance=dict(exporter_sha256=base.sha(__file__), torch=torch.__version__)))
    render_and_finish(output)


if __name__ == '__main__':
    main()
