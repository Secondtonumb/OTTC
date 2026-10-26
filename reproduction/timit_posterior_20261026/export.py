"""Audio-only, full-vocabulary posterior export from completed matched runs."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ottc-posterior-mpl')
import numpy as np
import torch
from reproduction.timit_ssl_native10_20261019.common import (
    ROOT, build_model, Collator, load_data, sha, write,
)

HERE = Path(__file__).resolve().parent
METHODS = [('ce', 'CE'), ('jsd_standard', 'JSD'), ('ag', 'AG-DIV'), ('jdiv', 'J-DIV')]


def render_and_finish(output):
    from reproduction.timit_posterior_20261026.plot import draw
    # Verify the interchange formats against the binary source before delivery.
    manifest = json.loads((output / 'manifest.json').read_text())
    for method in manifest['methods']:
        cost = method['cost']
        for name, digest in method['files'].items():
            assert sha(output / name) == digest, name
        with np.load(output / f'{cost}.npz') as arrays:
            payload = json.loads((output / f'{cost}.json').read_text())
            assert np.array_equal(arrays['posterior'], np.asarray(payload['posterior'], dtype=np.float32))
            assert np.allclose(arrays['posterior'].sum(-1), 1, atol=2e-6)
            with (output / f'{cost}.csv').open() as stream:
                records = list(csv.DictReader(stream))
            assert len(records) == arrays['posterior'].size
            for rec in records:
                assert float(rec['posterior']) == float(arrays['posterior'][int(rec['frame']), int(rec['phone_id'])])
    draw(output)
    write(output / 'COMPLETE.json', dict(methods=[k for k, _ in METHODS], real_audio=True,
          posterior_verified=True, npz_json_csv_identical=True,
          rendering_source_sha256=sha(HERE / 'plot.py'), exporter_sha256=sha(__file__),
          files={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
    print('COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=43, choices=[43, 44, 45])
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
    models_root = ROOT / 'reproduction/timit_seed_sweep_20261022/models'
    config = json.loads((models_root / f'ce__seed{args.seed}/run_config.json').read_text())
    _, rows, vocab, processor, data_info = load_data(config)
    row = next(r for r in rows['dev'] if r['utt'] == args.utt)
    original = next(json.loads(line) for line in (ROOT / 'reproduction/multihead_timit/data/train-dev.jsonl').open()
                    if json.loads(line)['utt'] == args.utt)
    assert row['wav'] == original['wav']
    labels = [p for p, i in sorted(vocab.items(), key=lambda item: item[1])]
    batch = Collator(processor)([dict(wav=row['wav'], labels=row['training_target_ids'])])
    batch.pop('labels')
    batch = {k: v.to(args.device) for k, v in batch.items()}
    model, topo = build_model(config, vocab, 73)
    assert topo is None
    model.to(device=args.device, dtype=torch.float32).eval()
    shutil.copyfile(row['wav'], output / 'audio.wav')
    write(output / 'vocab.json', vocab)
    write(output / 'reference.json', dict(utt=row['utt'], text=original['text'], split='dev',
          duration=row['duration'], references=original['references'],
          note='Mapped human PHN intervals copied without snapping, merging, or gap filling.'))
    exports = []
    for cost, title in METHODS:
        begin = time.monotonic()
        source = models_root / f'{cost}__seed{args.seed}'
        current = json.loads((source / 'run_config.json').read_text())
        assert (source / 'TRAIN_COMPLETE.json').exists() and (source / 'BENCHMARK_COMPLETE.json').exists()
        for key in ('backbone', 'head', 'frame_ms', 'native_conv_stride', 'forward_type', 'seed',
                    'global_batch', 'epochs', 'learning_rate', 'warmup_steps', 'weight_decay', 'data'):
            assert current[key] == config[key], key
        best = json.loads((source / 'BEST_DEV.json').read_text())
        checkpoint = (source / 'best_dev.pt').resolve()
        assert checkpoint == Path(best['path']).resolve()
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == best['step']
        assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
        model.load_state_dict(saved['model'], strict=True)
        model.step = saved['step']
        # With labels absent, the inference branch is independent of the cost/transport.
        # Each posterior difference comes from independently TRAINED weights.
        with torch.inference_mode():
            out = model(**batch)
        length = int(out.hidden_states[1][0].sum())
        logits = out.logits[0, :length].float().cpu().numpy()
        posterior = torch.from_numpy(logits).softmax(-1).numpy()
        mass = out.hidden_states[0][0, :length].float().cpu().numpy()
        frame_time = np.arange(length, dtype=np.float32) * np.float32(.01)
        # Actual CNN receptive field, independent of the historical left-grid clock.
        receptive_field, jump = 1, 1
        for kernel, stride in zip(model.wav2vec2.config.conv_kernel, current['native_conv_stride']):
            receptive_field += (kernel - 1) * jump
            jump *= stride
        centers = frame_time + np.float32((receptive_field - 1) / (2 * 16000))
        assert posterior.shape == (length, 42) and np.isfinite(posterior).all()
        assert np.allclose(posterior.sum(-1), 1, atol=2e-6)
        assert np.isclose(mass.sum(), 1, atol=2e-6)
        argmax = posterior.argmax(-1)
        historical = next(json.loads(line) for line in (source / 'benchmark_best_dev_dev.predictions.jsonl').open()
                          if json.loads(line)['utt'] == row['utt'])
        old_ids = np.asarray(historical['text_free']['frame_token_ids'])
        assert old_ids.shape == argmax.shape
        agreement = float((old_ids == argmax).mean())
        assert agreement >= .99, (cost, agreement)
        np.savez_compressed(output / f'{cost}.npz', posterior=posterior, logits=logits,
                            frame_time_seconds=frame_time, receptive_center_seconds=centers,
                            ot_frame_mass=mass, argmax_ids=argmax, labels=np.asarray(labels))
        payload = dict(method=title, utterance=row['utt'], labels=labels,
                       frame_time_seconds=frame_time.tolist(), receptive_center_seconds=centers.tolist(),
                       posterior=posterior.tolist(), logits=logits.tolist(),
                       ot_frame_mass=mass.tolist(), argmax_ids=argmax.tolist())
        write(output / f'{cost}.json', payload)
        with (output / f'{cost}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame', 'time_seconds', 'receptive_center_seconds', 'phone_id', 'phone', 'posterior', 'logit', 'ot_frame_mass'])
            for t in range(length):
                for p in range(42):
                    writer.writerow([t, float(frame_time[t]), float(centers[t]), p, labels[p],
                                     float(posterior[t, p]), float(logits[t, p]), float(mass[t])])
        exports.append(dict(cost=cost, title=title, seed=args.seed, shape=list(posterior.shape),
                            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                            config_sha256=sha(source / 'run_config.json'), epoch=saved['epoch'], step=saved['step'],
                            saved_fp32=True, training_complete=True, selection=best['selection'],
                            historical_argmax_agreement=agreement, seconds=time.monotonic()-begin,
                            files={f'{cost}.{ext}': sha(output / f'{cost}.{ext}') for ext in ['npz', 'json', 'csv']}))
        del saved, out
        gc.collect()
        print('EXPORTED', cost, exports[-1], flush=True)
    write(output / 'manifest.json', dict(utterance=row['utt'], split='dev', text=original['text'],
          selection='fixed first dev utterance; not selected for visual model differences',
          source_audio=row['wav'], audio_sha256=sha(output / 'audio.wav'), duration=row['duration'],
          data=data_info, methods=exports, labels=labels,
          inference=dict(device=args.device, dtype='float32', autocast=False, eval=True, labels_supplied=False,
                         posterior='softmax(lm_head(hidden), dim=phone)', ot_frame_mass='separate lm_weight softmax over time',
                         reference_used_for_inference=False, optimizer_updated=False,
                         clock='frame_time_seconds = frame_index * 0.01, historical native-grid convention',
                         receptive_field_samples=receptive_field, receptive_center_offset_seconds=(receptive_field-1)/(2*16000),
                         no_resampling_no_smoothing_no_temperature=True),
          missing_methods=dict(kl='No completed matched checkpoint in this sweep; omitted per user instruction.'),
          provenance=dict(script_sha256=sha(__file__), torch=torch.__version__,
                          model_source_sha256=sha(ROOT / 'ottc/models/sequence/wav2vec2_ottc.py'))))
    render_and_finish(output)


if __name__ == '__main__':
    main()
