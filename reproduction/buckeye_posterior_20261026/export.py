"""Export an audio-only posterior example from pinned Buckeye best-dev weights."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/ottc-buckeye-posterior-mpl')
import numpy as np
import torch
from transformers import Wav2Vec2Processor

from reproduction.buckeye_dataset import load_buckeye_rows, resolve_buckeye
from reproduction.timit_ssl_native10_20261019.common import ROOT, build_model, Collator, sha, write

HERE = Path(__file__).resolve().parent
METHODS = [('ce', 'CE'), ('jsd_standard', 'JSD'), ('ag', 'AG-DIV'), ('jdiv', 'J-DIV')]
FAMILY = ROOT / 'reproduction/buckeye_full40_seed_sweep_20261025'


def summarize(posterior, labels):
    p = posterior.astype(np.float64)
    ranked = np.sort(p, axis=-1)
    ids = p.argmax(-1)
    return dict(frames=len(p), classes=len(labels),
                mean_max_posterior=float(ranked[:, -1].mean()),
                mean_entropy_nats=float(-(p * np.log(np.maximum(p, 1e-300))).sum(-1).mean()),
                mean_top1_top2_margin=float((ranked[:, -1] - ranked[:, -2]).mean()),
                argmax_switches=int(np.count_nonzero(np.diff(ids))),
                per_class=[dict(phone=label, mean_posterior=float(p[:, i].mean()),
                                max_posterior=float(p[:, i].max()), argmax_frames=int((ids == i).sum()))
                           for i, label in enumerate(labels)])


def finish(output):
    from reproduction.buckeye_posterior_20261026.render import draw
    manifest = json.loads((output / 'manifest.json').read_text())
    statistics = {}
    for method in manifest['methods']:
        cost = method['cost']
        for filename, digest in method['files'].items():
            assert sha(output / filename) == digest
        with np.load(output / f'{cost}.npz') as arrays:
            p = arrays['posterior']
            raw = json.loads((output / f'{cost}.json').read_text())
            assert np.array_equal(p, np.asarray(raw['posterior'], dtype=np.float32))
            assert np.allclose(p.sum(-1), 1, atol=2e-6)
            with (output / f'{cost}.csv').open() as stream:
                records = list(csv.DictReader(stream))
            assert len(records) == p.size
            for record in records:
                assert float(record['posterior']) == float(p[int(record['frame']), int(record['phone_id'])])
            statistics[cost] = summarize(p, manifest['labels'])
    write(output / 'statistics.json', dict(scope='one fixed dev utterance; descriptive only; not corpus statistics or a loss ranking',
          entropy_units='nats; all 42 IDs; no renormalization', methods=statistics))
    draw(output)
    write(output / 'COMPLETE.json', dict(export_complete=True,
          training_complete=all(m['training_complete'] for m in manifest['methods']),
          scope='posterior export completion only; training and benchmark status are separate',
          posterior_verified=True, npz_json_csv_identical=True, evaluation_uploads=False,
          files={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}))
    print('EXPORT_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=43, choices=[42, 43, 44])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--render-only', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.render_only:
        finish(args.render_only)
        return
    data_root, prepared = resolve_buckeye()
    audit = json.loads((data_root / 'audit.json').read_text())
    rows, vocab, contract = load_buckeye_rows('dev', training_speakers=audit['splits']['train']['speakers'])
    row = rows[0]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = HERE / f'seed{args.seed}' / row['utt_id'] / stamp
    output.mkdir(parents=True, exist_ok=False)
    labels = [label for label, i in sorted(vocab.items(), key=lambda item: item[1])]
    assert len(labels) == 42 and prepared['split_speaker_and_track_disjoint']
    configs, snapshots, handles = {}, {}, {}
    # Pin all four files before loading the model. Open descriptors survive an
    # active trainer's retention unlink; no checkpoint or resume contract changes.
    for cost, _ in METHODS:
        source = FAMILY / 'models' / f'{cost}__seed{args.seed}'
        configs[cost] = json.loads((source / 'run_config.json').read_text())
        best_text = (source / 'BEST_DEV.json').read_text()
        best = json.loads(best_text)
        handle = Path(best['path']).open('rb')
        handles[cost] = handle
        snapshots[cost] = dict(best=best, source=source,
                               training_complete=(source / 'TRAIN_COMPLETE.json').exists(),
                               benchmark_complete=(source / 'BENCHMARK_COMPLETE.json').exists())
        write(output / f'{cost}.BEST_DEV.snapshot.json', best)
    config = configs['ce']
    for cost, _ in METHODS:
        current = configs[cost]
        assert current['expected_data']['dev'] == contract
        for key in ('backbone', 'head', 'frame_ms', 'native_conv_stride', 'forward_type', 'seed',
                    'global_batch', 'epochs', 'learning_rate', 'warmup_steps', 'weight_decay', 'expected_data'):
            assert current[key] == config[key], key
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    batch = Collator(processor)([dict(wav=row['wav'], labels=row['training_target_ids'])])
    batch.pop('labels')
    model, topo = build_model(config, vocab, 285)
    assert topo is None
    model.to(device='cpu', dtype=torch.float32).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    shutil.copyfile(row['wav'], output / 'audio.wav')
    write(output / 'vocab.json', vocab)
    write(output / 'reference.json', dict(utt=row['utt_id'], source_utt=row['utt'], text=row['wrd'],
          split='dev', duration=row['duration'], speaker=row['speaker'], words=row['words'],
          references=dict(ottc_arpabet=row['reference'], native=row['reference_native']),
          reference_source_rows=row['reference_source_rows'],
          note='Exact full40 gold intervals; no quantization, merging, artificial silence or gap filling.'))
    exports = []
    for cost, title in METHODS:
        snap, handle = snapshots[cost], handles[cost]
        best = snap['best']
        hasher = hashlib.sha256()
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        assert digest == best['sha256'], cost
        saved = torch.load(f'/proc/self/fd/{handle.fileno()}', map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == best['step'] and saved['epoch'] == best['epoch']
        assert saved['recipe'] == configs[cost]['name'] and saved['seed'] == args.seed
        assert saved['source_sha256'] == configs[cost]['source_sha256']
        assert all(t.dtype == torch.float32 for t in saved['model'].values() if t.is_floating_point())
        model.load_state_dict(saved['model'], strict=True)
        model.step = saved['step']
        with torch.inference_mode():
            out = model(**batch)
        length = int(out.hidden_states[1][0].sum())
        logits = out.logits[0, :length].float().cpu().numpy()
        posterior = torch.from_numpy(logits).softmax(-1).numpy()
        mass = out.hidden_states[0][0, :length].float().cpu().numpy()
        receptive_field, hop = 1, 1
        for kernel, stride in zip(model.wav2vec2.config.conv_kernel, config['native_conv_stride']):
            receptive_field += (kernel - 1) * hop
            hop *= stride
        assert hop == 160
        times = np.arange(length, dtype=np.float32) * np.float32(.01)
        centers = times + np.float32((receptive_field - 1) / (2 * 16000))
        ids = posterior.argmax(-1)
        assert posterior.shape == (length, 42) and np.isfinite(posterior).all()
        assert np.allclose(posterior.sum(-1), 1, atol=2e-6) and np.isclose(mass.sum(), 1, atol=2e-6)
        np.savez_compressed(output / f'{cost}.npz', posterior=posterior, logits=logits,
                            frame_time_seconds=times, receptive_center_seconds=centers,
                            ot_frame_mass=mass, argmax_ids=ids, labels=np.asarray(labels))
        write(output / f'{cost}.json', dict(method=title, utterance=row['utt_id'], labels=labels,
              frame_time_seconds=times.tolist(), receptive_center_seconds=centers.tolist(),
              posterior=posterior.tolist(), logits=logits.tolist(), ot_frame_mass=mass.tolist(), argmax_ids=ids.tolist()))
        with (output / f'{cost}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame', 'time_seconds', 'receptive_center_seconds', 'phone_id', 'phone', 'posterior', 'logit', 'ot_frame_mass'])
            for t in range(length):
                for p in range(42):
                    writer.writerow([t, float(times[t]), float(centers[t]), p, labels[p],
                                     float(posterior[t, p]), float(logits[t, p]), float(mass[t])])
        exports.append(dict(cost=cost, title=title, seed=args.seed, shape=list(posterior.shape),
                            checkpoint=best['path'], checkpoint_sha256=digest,
                            config_sha256=sha(snap['source'] / 'run_config.json'),
                            epoch=saved['epoch'], step=saved['step'], best_dev_PER=best['dev_PER'],
                            saved_fp32=True, training_complete=snap['training_complete'],
                            benchmark_complete=snap['benchmark_complete'],
                            selection='current single best Buckeye dev PER checkpoint at export start; interim snapshot',
                            files={f'{cost}.{ext}': sha(output / f'{cost}.{ext}') for ext in ('npz', 'json', 'csv')}))
        print('EXPORTED', cost, 'epoch', saved['epoch'], 'shape', posterior.shape, flush=True)
        del saved, out
        handle.close()
        gc.collect()
    write(output / 'manifest.json', dict(utterance=row['utt_id'], split='dev', text=row['wrd'], seed=args.seed,
          duration=row['duration'], source_audio=row['wav'], audio_sha256=sha(output / 'audio.wav'),
          snapshot_utc=stamp, selection='fixed first full40 dev utterance; not selected for visual differences',
          data=contract, labels=labels, methods=exports, evaluation_uploads=False,
          inference=dict(device='cpu', dtype='float32', autocast=False, eval=True, labels_supplied=False,
                         reference_used_for_inference=False, optimizer_updated=False,
                         posterior='softmax(logits, dim=phone); all 42 IDs',
                         ot_frame_mass='separate mass-head softmax over time; not phone posterior',
                         frame_hop_samples=hop, receptive_field_samples=receptive_field,
                         receptive_center_offset_seconds=(receptive_field - 1) / (2 * 16000),
                         no_resampling_no_smoothing_no_temperature=True),
          provenance=dict(exporter_sha256=sha(__file__), torch=torch.__version__,
                          model_source_sha256=sha(ROOT / 'ottc/models/sequence/wav2vec2_ottc.py'),
                          dataset_registry_sha256=sha(ROOT / 'reproduction/dataset_defaults.json'))))
    finish(output)
    write(HERE / 'LATEST_EXPORT.json', dict(output=str(output), snapshot_utc=stamp))


if __name__ == '__main__':
    main()
