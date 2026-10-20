"""Full-set audio-only alpha decoding. Scoring references are never opened here."""
import gc
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers

transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import Wav2Vec2Processor
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from reproduction.alignment_decoders import native_ot_spans
from reproduction.ottc_alpha_full_20261017.prepare import ROOT, HERE, digest, write


def load_existing(path):
    rows = {}
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            assert row['recording_id'] not in rows
            rows[row['recording_id']] = row
    return rows


@torch.no_grad()
def main():
    assert torch.cuda.is_available(), 'Use an allocated GPU'
    torch.set_num_threads(4)
    source_paths = [Path(__file__), HERE / 'prepare.py', HERE / 'INPUTS.json',
        ROOT / 'reproduction/alignment_decoders.py', ROOT / 'ottc/models/sequence/wav2vec2_ottc.py',
        ROOT / 'ottc/ops_loss/ot_loss.py']
    contract = dict(source_sha256={str(p): digest(p) for p in source_paths},
        precision='FP32; batch_size=1; same original inference', frame_seconds=.02,
        beta='1 / own greedy predicted phone count; no reference input',
        methods=['argmax', 'alpha_grid', 'alpha_fractional'])
    contract_path = HERE / 'DECODE_CONTRACT.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract, 'Active source/input contract changed'
    else:
        write(contract_path, contract)
    if (HERE / 'DECODE_COMPLETE.json').exists():
        print('Reuse completed full evaluation', flush=True)
        return
    inputs = json.loads((HERE / 'INPUTS.json').read_text())
    started = time.monotonic()
    summaries = []
    for job in inputs['jobs']:
        model_done = HERE / (job['dataset'] + '_COMPLETE.json')
        output = HERE / (job['dataset'] + '_predictions.jsonl')
        existing = load_existing(output)
        expected = {c['recording_id'] for c in job['cases']}
        assert set(existing) <= expected
        if model_done.exists():
            assert set(existing) == expected
            saved = json.loads(model_done.read_text())
            assert saved['predictions_sha256'] == digest(output)
            summaries.append(saved)
            continue
        checkpoint = Path(job['checkpoint'])
        weights = checkpoint / 'model.safetensors'
        before = weights.stat()
        assert digest(weights) == job['checkpoint_sha256']
        model, loading = wvForOTTC.from_pretrained(checkpoint, output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
        assert model.config.bins_metric == 'jsd'
        model.step = int(checkpoint.name.split('-')[-1])
        model.eval().cuda()
        processor = Wav2Vec2Processor.from_pretrained(checkpoint.parent / 'processor')
        vocab = json.loads((checkpoint.parent / 'vocab_used.json').read_text())
        inverse = {int(v): p for p, v in vocab.items()}
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        torch.cuda.reset_peak_memory_stats()
        job_start = time.monotonic()
        print('MODEL START', job['dataset'], 'already decoded', len(existing), '/', len(expected), flush=True)
        with output.open('a', buffering=1) as stream:
            for i, case in enumerate(job['cases']):
                rid = case['recording_id']
                if rid in existing:
                    continue
                case_start = time.monotonic()
                audio_bytes = Path(case['wav']).read_bytes()
                import hashlib
                audio_hash = hashlib.sha256(audio_bytes).hexdigest()
                wave, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                assert sr == 16000 and wave.ndim == 1
                assert abs(len(wave) / sr - case['duration']) < 1e-8
                batch = processor(wave, sampling_rate=sr, return_tensors='pt', return_attention_mask=True).to('cuda')
                out = model(**batch)
                frames = int(out.hidden_states[1][0].sum())
                logits = out.logits[0, :frames].float().cpu().numpy()
                alpha = out.hidden_states[0][0, :frames].float().cpu().numpy().astype(np.float64)
                assert np.isfinite(alpha).all() and np.all(alpha >= 0) and alpha.sum() > 0
                ids = logits.argmax(-1)
                ids[alpha == 0] = 0
                ids[ids == 41] = 0
                hypothesis = processor.decode(ids).split()
                target_ids = [vocab[p] for p in hypothesis]
                assert all(v not in (0, 41) for v in target_ids)
                # Saved audio-only predictions are checked only after the independent forward.
                prior = case['prior_audio_only_prediction']
                assert hypothesis == prior['hypothesis'], ('hypothesis replay mismatch', rid)
                if 'frame_token_ids' in prior:
                    assert ids.tolist() == prior['frame_token_ids'], ('frame replay mismatch', rid)
                changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
                if 'spans' in prior:
                    starts, ends = np.r_[0, changes], np.r_[changes, len(ids)]
                    spans = [[inverse[int(ids[s])], float(s) * .02, min(float(e) * .02, case['duration'])]
                             for s, e in zip(starts, ends) if ids[s] != 0]
                    assert spans == prior['spans'], ('span replay mismatch', rid)
                    assert [int(v) * .02 for v in changes if .02 < int(v) * .02 < case['duration'] - .02] == prior['boundaries']
                mass = alpha / alpha.sum()
                U = len(target_ids)
                native = native_ot_spans(mass, target_ids) if U else []
                positions = np.asarray([s[2] for s in native[:-1]], dtype=np.float64)
                cdf = np.r_[0., mass.cumsum()]
                cdf[-1] = 1.
                if len(positions):
                    cells = np.minimum(np.floor(positions).astype(int), frames - 1)
                    crossings = cdf[cells] + (positions - cells) * mass[cells]
                    np.testing.assert_allclose(crossings, np.arange(1, U) / U, atol=1e-12, rtol=0)
                    assert np.all(np.diff(positions) > 0)
                # Dense marginal verification only on the three preselected size probes per model.
                dense_verified = False
                if i < 3 and U:
                    target_cdf = np.linspace(0., 1., U + 1)
                    gamma = np.maximum(0., np.minimum(cdf[1:, None], target_cdf[None, 1:]) -
                                       np.maximum(cdf[:-1, None], target_cdf[None, :-1]))
                    np.testing.assert_allclose(gamma.sum(1), mass, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(gamma.sum(0), np.ones(U) / U, rtol=0, atol=1e-12)
                    dense_verified = True
                    del gamma
                rounded = np.floor(positions + .5)
                shifts = np.abs(positions - rounded) * 20
                record = dict(recording_id=rid, utt=case['utt'], split=case['split'], dataset=job['dataset'],
                    duration=case['duration'], wav_sha256=audio_hash, hypothesis=hypothesis, hypothesis_token_ids=target_ids,
                    predicted_phone_count=U, frame_count=frames, frame_token_ids=ids.tolist(), alpha=mass.tolist(),
                    argmax_boundaries=(changes * .02).tolist(),
                    native_alpha_boundaries=(positions * .02).tolist(),
                    native_alpha_grid_boundaries=(rounded * .02).tolist(),
                    fractional_boundary_frame_positions=positions.tolist(),
                    sum_abs_subframe_shift_ms=float(shifts.sum()), subframe_boundary_count=len(shifts),
                    max_abs_subframe_shift_ms=float(shifts.max()) if len(shifts) else 0.,
                    rounded_boundary_collisions=len(rounded) - len(set(rounded)), empty_hypothesis=not bool(U),
                    recognition_replay_identical=True, quantile_crossings_verified=True,
                    dense_transport_marginals_verified=dense_verified,
                    reference_phone_count_used=False, reference_phone_labels_used=False, reference_times_used=False,
                    text_free=True, seconds=time.monotonic() - case_start)
                stream.write(json.dumps(record) + '\n')
                existing[rid] = record
                if i < 3 or (i + 1) % 100 == 0 or len(existing) == len(expected):
                    progress = dict(dataset=job['dataset'], completed=len(existing), expected=len(expected),
                        last_split=case['split'], last_utt=case['utt'], last_duration=case['duration'],
                        last_seconds=record['seconds'], elapsed_seconds=time.monotonic() - job_start,
                        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                        host=platform.node(), pbs_job=os.environ.get('PBS_JOBID'))
                    write(HERE / 'PROGRESS.json', progress)
                    print('PROGRESS', json.dumps(progress), flush=True)
                del out, batch
        assert set(existing) == expected
        after = weights.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        assert digest(weights) == job['checkpoint_sha256']
        summary = dict(dataset=job['dataset'], checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
            original_epoch=state['epoch'], original_step=state['global_step'], utterances=len(expected), loading_info=loading,
            checkpoint_unchanged=True, seconds=time.monotonic() - job_start,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30, predictions_sha256=digest(output),
            empty_hypotheses=sum(c['empty_hypothesis'] for c in existing.values()),
            recognition_replay_identical=True, host=platform.node(), pbs_job=os.environ.get('PBS_JOBID'))
        write(model_done, summary)
        summaries.append(summary)
        del model, existing
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(m['utterances'] for m in summaries) == inputs['expected_utterances']
    for path, sha in contract['source_sha256'].items():
        assert digest(path) == sha, ('source changed during decoding', path)
    write(HERE / 'DECODE_COMPLETE.json', dict(status='complete', scope=inputs['scope'], utterances=7590,
        expected_by_split=inputs['expected_by_split'], models=summaries, contract=contract,
        reference_inputs_used=False, new_training=False, optimizer_updates=0,
        wall_seconds=time.monotonic() - started, host=platform.node(), pbs_job=os.environ.get('PBS_JOBID')))
    print('DECODE COMPLETE 7590', flush=True)


if __name__ == '__main__':
    main()
