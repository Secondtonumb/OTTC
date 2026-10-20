"""K2 phone-reference FA of the exact early TIMIT CE/JSD weights."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'reproduction/vendor'), str(ROOT / 'reproduction'), str(ROOT)]
import numpy as np
import soundfile as sf
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import Wav2Vec2Processor
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from ifmdd_k2_alignment import load_backend, align
from segmentation_metrics import r_value_percent, SOURCE as R_SOURCE

EXPERIMENT = Path(__file__).resolve().parent
SPLIT = 'timit_core_test_192'
MODELS = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60'
OLD_RESULTS = ROOT / 'reproduction/results'
OBJECTIVES = {'ce': 'runexp_ottc_fuse_phoneme', 'jsd_upstream': 'runexp_ottc_fuse_phoneme_jsd',
              'jsd_standard': 'runexp_ottc_fuse_phoneme_jsd_standard'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def matches(ref, pred, tolerance):
    i = j = n = 0
    while i < len(ref) and j < len(pred):
        if abs(ref[i] - pred[j]) <= tolerance + 1e-9:
            n += 1
            i += 1
            j += 1
        elif pred[j] < ref[i]:
            j += 1
        else:
            i += 1
    return n


def boundaries(spans, duration):
    return sorted({float(t) for _, s, e in spans for t in (s, e) if .02 < t < duration - .02})


def publish(result, output, smoke):
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    options = dict(project=defaults['project'], entity=defaults['entity'],
                   name=f"singlehead-{result['objective']}-K2FA-core192" + ('-smoke' if smoke else ''),
                   group='timit_singlehead_k2_fa_20261013', job_type='evaluation',
                   tags=['single-head', 'phone-reference-FA', 'k2', 'smoke' if smoke else 'full'],
                   dir=str(output), mode=os.environ.get('WANDB_MODE', defaults['mode']),
                   config={k: result[k] for k in ['objective', 'checkpoint', 'checkpoint_sha256', 'protocol']},
                   settings=wandb.Settings(init_timeout=45, disable_code=True))
    try:
        run = wandb.init(**options)
    except wandb.errors.CommError:
        options['mode'] = 'offline'
        run = wandb.init(**options)
    (output / 'wandb_run.json').write_text(json.dumps(dict(id=run.id, url=run.url,
        project=defaults['project'], entity=defaults['entity'], mode=run.settings.mode), indent=2) + '\n')
    values = {}
    for ms, metrics in result['forced_k2']['boundaries'].items():
        for key, value in metrics.items():
            values[f'benchmark/core192/phone_reference_k2/{key}_{ms}ms'] = value
    for key, value in result['forced_k2'].items():
        if isinstance(value, (int, float)):
            values[f'benchmark/core192/phone_reference_k2/{key}'] = value
    run.log(values)
    run.summary.update(values)
    run.summary['evaluation_only'] = True
    run.summary['optimizer_updated'] = False
    artifact = wandb.Artifact(f"singlehead-{result['objective']}-k2-core192" + ('-smoke' if smoke else ''),
                             type='evaluation')
    for name in ['metrics.json', 'predictions.jsonl', 'ifmdd_k2_source_snapshot.py', 'evaluator_snapshot.py']:
        artifact.add_file(str(output / name), name=name)
    run.log_artifact(artifact)
    run.finish()


def evaluate(objective, limit):
    begin = time.monotonic()
    suffix = OBJECTIVES[objective]
    model_dir = MODELS / suffix
    checkpoint = model_dir / 'checkpoint-2920'
    source = OLD_RESULTS / suffix / (SPLIT + '.jsonl')
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    other_suffix = OBJECTIVES['jsd_upstream' if objective == 'ce' else 'ce']
    other = [json.loads(line) for line in (OLD_RESULTS / other_suffix / (SPLIT + '.jsonl')).read_text().splitlines()]
    assert len(rows) == len(other) == 192
    assert len({r['utt'] for r in rows}) == 192
    assert all((a['utt'], a['wav'], a['reference']) == (b['utt'], b['wav'], b['reference'])
               for a, b in zip(rows, other))
    if limit:
        rows = rows[:limit]
    output = EXPERIMENT / ('smoke' if limit else 'full') / objective
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'EVAL_COMPLETE.json').exists(), 'Preserve a completed evaluation'
    backend, provenance = load_backend()
    (output / 'ifmdd_k2_source_snapshot.py').write_text(provenance.pop('source_snapshot') + '\n')
    (output / 'evaluator_snapshot.py').write_bytes(Path(__file__).read_bytes())
    vocab = json.loads((model_dir / 'vocab_used.json').read_text())
    inverse = {v: k for k, v in vocab.items()}
    assert all(p in vocab and vocab[p] not in (0, 41) for r in rows for p, _, _ in r['reference'])
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['epoch'] == 40 and state['global_step'] == 2920
    model, loading = wvForOTTC.from_pretrained(checkpoint, output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), loading
    expected_cost = {'ce': 'crossentropy', 'jsd_upstream': 'jsd', 'jsd_standard': 'jsd_standard'}
    assert model.config.bins_metric == expected_cost[objective]
    assert model.config.vocab_size == 42
    model.eval().cuda()
    model.step = 2920
    processor = Wav2Vec2Processor.from_pretrained(model_dir / 'processor')
    tokenizer_vocab = processor.tokenizer.get_vocab()
    assert all(tokenizer_vocab[p] == token for p, token in vocab.items())
    assert all(token >= 42 for p, token in tokenizer_vocab.items() if p not in vocab)
    counts = {str(ms): Counter() for ms in (10, 20, 30)}
    errors = Counter()
    gaps = 0
    with (output / 'predictions.jsonl').open('w') as f, torch.inference_mode():
        for i, row in enumerate(rows):
            audio, sr = sf.read(row['wav'], dtype='float32')
            assert sr == 16000 and audio.ndim == 1
            duration = len(audio) / sr
            values = processor(audio, sampling_rate=sr, return_tensors='pt', return_attention_mask=True).to('cuda')
            out = model(**values)  # Never pass labels/reference timestamps to the model.
            logits = out.logits[0].float()
            mask = out.hidden_states[1][0]
            assert int(mask.sum()) == len(logits)
            assert torch.isfinite(logits).all()
            ids = logits.argmax(-1).cpu().numpy()
            ids[out.hidden_states[0].cpu().numpy().reshape(-1) == 0] = 0
            ids[ids == 41] = 0
            changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            starts, ends = np.r_[0, changes], np.r_[changes, len(ids)]
            replay_spans = [[inverse[int(ids[s])], float(s) * .02, min(float(e) * .02, duration)]
                            for s, e in zip(starts, ends) if ids[s] != 0]
            assert processor.decode(ids).split() == row['hypothesis'], ('recognition replay', row['utt'])
            assert replay_spans == row['predicted_spans'], ('frame segmentation replay', row['utt'])
            assert [int(t) * .02 for t in changes if .02 < int(t) * .02 < duration - .02] == row['predicted_boundaries']
            reference = row['reference']
            targets = [vocab[p] for p, _, _ in reference]
            segments, path = align(logits.log_softmax(-1), targets, duration, backend, blank_policy='previous')
            spans = [[inverse[s['token_id']], s['start'], s['end']] for s in segments]
            assert [p for p, _, _ in spans] == [p for p, _, _ in reference]
            rb = boundaries(reference, duration)
            pb = boundaries(spans, duration)
            assert rb == row['reference_boundaries']
            for ms, count in counts.items():
                count['matches'] += matches(rb, pb, int(ms) / 1000)
                count['reference'] += len(rb)
                count['predicted'] += len(pb)
            for (_, rs, re), (_, ps, pe) in zip(reference, spans):
                errors['tokens'] += 1
                errors['start_abs'] += abs(rs - ps)
                errors['end_abs'] += abs(re - pe)
                errors['start_hit20'] += abs(rs - ps) <= .02 + 1e-9
                errors['end_hit20'] += abs(re - pe) <= .02 + 1e-9
            gaps += sum(s1 - e0 > 1e-7 for (_, _, e0), (_, s1, _) in zip(reference, reference[1:]))
            f.write(json.dumps(dict(utt=row['utt'], wav=row['wav'], duration=duration,
                reference=reference, supplied_target_ids=targets, predicted_spans=spans,
                k2_segments=segments, frame_token_ids=path, reference_boundaries=rb,
                predicted_boundaries=pb, frames=len(logits), text_free_replay_identical=True,
                reference_times_used_by_aligner=False)) + '\n')
            if (i + 1) % 24 == 0 or i + 1 == len(rows):
                print(objective, i + 1, '/', len(rows), f'{time.monotonic()-begin:.1f}s', flush=True)
    metrics = dict(utterances=len(rows), alignment_failures=0, aligned_tokens=errors['tokens'], boundaries={})
    for ms, c in counts.items():
        m, r, p = c['matches'], c['reference'], c['predicted']
        precision, recall = 100 * m / p, 100 * m / r
        metrics['boundaries'][ms] = dict(c, precision_percent=precision, recall_percent=recall,
            f1_percent=200 * m / (r + p), r_value_percent=r_value_percent(precision, recall))
    n = errors['tokens']
    metrics.update(start_MAE_ms=1000*errors['start_abs']/n, end_MAE_ms=1000*errors['end_abs']/n,
        boundary_MAE_ms=500*(errors['start_abs']+errors['end_abs'])/n,
        start_plus_end_TSE_ms=1000*(errors['start_abs']+errors['end_abs'])/n,
        start_hit20_percent=100*errors['start_hit20']/n, end_hit20_percent=100*errors['end_hit20']/n)
    result = dict(objective=objective, checkpoint=str(checkpoint), epoch=40, global_step=2920,
        checkpoint_sha256=sha256(checkpoint / 'model.safetensors'), checkpoint_loading_info=loading,
        evaluator_sha256=sha256(Path(__file__)), original_prediction_manifest=str(source),
        original_prediction_manifest_sha256=sha256(source),
        protocol=dict(mode='supplied-phone-reference K2 forced alignment', audio_only_model_inference=True,
            reference_times_used_by_aligner=False, frame_stride_seconds=.02,
            logits='Native 42-class FP32 posterior; no subphone collapse or learned frame-mass use',
            supplied_phones='Exact old core192 reference sequence including sil; dx mapped to d, q removed by historical manifest',
            scoring='Unique interval endpoints; exclude first/last20ms; one-to-one time-only matching at10/20/30ms',
            reference_gap_count=gaps, reference_gaps='Preserve historical endpoints around omitted q; do not fill reference gaps',
            timestamp_grid='duration / frames; no offset', blank_policy='previous',
            k2=provenance, r_value_source=str(R_SOURCE), r_value_source_sha256=sha256(R_SOURCE),
            evaluation_only=True, optimizer_updated=False, word_transcript_FA_Bench=False),
        text_free_replay=dict(utterances_checked=len(rows), identical_hypotheses=len(rows),
                             identical_spans=len(rows), identical_boundaries=len(rows)),
        historical_text_free=json.loads((OLD_RESULTS / suffix / 'metrics.json').read_text())['splits'][SPLIT],
        forced_k2=metrics, elapsed_seconds=time.monotonic()-begin)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(objective, json.dumps(metrics), flush=True)
    publish(result, output, bool(limit))
    (output / 'EVAL_COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        utterances=len(rows), alignment_failures=0, elapsed_seconds=time.monotonic()-begin), indent=2) + '\n')
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--objectives', nargs='+', choices=list(OBJECTIVES), default=['ce', 'jsd_upstream'])
    args = p.parse_args()
    assert 0 <= args.limit <= 192
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'Use a GPU PBS allocation'
    for objective in args.objectives:
        evaluate(objective, args.limit)


if __name__ == '__main__':
    main()
