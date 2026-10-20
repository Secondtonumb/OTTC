"""TIMIT phone PER and text-free RNN-T emission-time boundary diagnostics."""
from collections import Counter
import json
from pathlib import Path

import torch

from reproduction.cif_timit_20261016.data import collate, read_audio, sha
from reproduction.cif_timit_20261016.evaluate import edits
from reproduction.segmentation_metrics import _scorer, SOURCE


def emission_boundaries(frames, duration, frame_count):
    # Interpret an emission as a phone-start proxy; discard the first phone's
    # start and utterance endpoints. Preserve coincident repeated emissions.
    return [f * duration / frame_count for f in frames[1:] if 0 < f < frame_count]


@torch.no_grad()
def evaluate(model, rows, vocab, output=None, autocast_dtype=None):
    if not rows:
        raise ValueError('empty evaluation split')
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts, records = Counter(), {t: Counter() for t in (10, 20, 30)}, []
    inverse = {v: k for k, v in vocab.items()}
    normalize = lambda seq: ['d' if p == 'dx' else p for p in seq if p not in ('sil', 'q')]
    try:
        for row in rows:
            batch = collate([row], device)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = model.recognize(batch['input_values'], batch['attention_mask'])
                supervised = model(**batch)
            hyp = [inverse[i] for i in prediction['tokens'][0]]
            reference = [p for p, _, _ in row['references']['ottc_arpabet']]
            duration = len(read_audio(row)) / 16000
            frame_count = int(prediction['frame_lengths'][0])
            predicted = emission_boundaries(prediction['emission_frames'][0], duration, frame_count)
            ref_times = [float(x[1]) for x in row['references']['ottc_arpabet'][1:] if 0 < float(x[1]) < duration]
            for tolerance, counter in counts.items():
                counter.update(_scorer.precision_recall_counts(predicted, ref_times, tolerance / 1000))
            totals.update(utterances=1, errors=edits(reference, hyp), phones=len(reference),
                normalized_errors=edits(normalize(reference), normalize(hyp)),
                normalized_phones=len(normalize(reference)), rnnt_sum=float(supervised['rnnt_sum']),
                ottc_sum=float(supervised['ottc_sum']), cap_hits=prediction['cap_hits'][0],
                cap_hit_utterances=int(prediction['cap_hits'][0] > 0))
            records.append(dict(utt=row['utt'], reference=reference, hypothesis=hyp,
                reference_boundaries=ref_times, emission_frames=prediction['emission_frames'][0],
                emission_boundaries=predicted, frame_count=frame_count, duration=duration,
                cap_hits=prediction['cap_hits'][0]))
    finally:
        model.train(was_training)
    result = dict(utterances=totals['utterances'], raw_PER_percent=100*totals['errors']/totals['phones'],
        normalized_nosil_PER_percent=100*totals['normalized_errors']/max(1, totals['normalized_phones']),
        rnnt_per_phone=totals['rnnt_sum']/totals['phones'],
        ottc_per_utterance=totals['ottc_sum']/totals['utterances'], counts=dict(totals),
        audio_only=True, forced_alignment=False, rvalue_source_sha256=sha(SOURCE),
        protocol=dict(prediction='greedy RNN-T emission frames [1:] as phone-start proxies; repeated phones/times retained',
            clock='emission frame index * audio duration / valid encoder frames; endpoints excluded',
            matching='frozen IF-MDD directional greedy pooled P/R/F1/R at 10/20/30ms; label_aware=False, lenient=False',
            reference='frozen TIMIT40 occurrence starts [1:]; utterance endpoints excluded',
            interpretation='emission timing diagnostic; RNN-T loss does not directly supervise phone boundaries',
            supervised_loss='teacher-forced references used only for validation loss, never recognition or emission timing'))
    result['text_free_emission'] = {str(t): _scorer.finalize_counts(c['precision_matches'], c['recall_matches'],
        c['predicted_boundaries'], c['reference_boundaries']) for t, c in counts.items()}
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
        Path(output).with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
    return result


def flat_metrics(prefix, result):
    out = {f'{prefix}/{k}': result[k] for k in ('raw_PER_percent', 'normalized_nosil_PER_percent',
        'rnnt_per_phone', 'ottc_per_utterance')}
    out[f'{prefix}/cap_hits'] = result['counts'].get('cap_hits', 0)
    for tolerance, metric in result['text_free_emission'].items():
        for key in ('precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct'):
            out[f'{prefix}/text_free_emission/{key}_{tolerance}ms'] = metric[key]
    return out
