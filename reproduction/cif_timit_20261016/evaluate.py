"""Audio-only CIF segmentation using frozen IF-MDD boundary P/R/F1/R-value."""
from collections import Counter
import json
from pathlib import Path

import torch

from reproduction.segmentation_metrics import _scorer, SOURCE
from reproduction.cif_timit_20261016.data import collate, read_audio, sha


def edits(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def predicted_boundaries(cif, duration, frame_count):
    """All internal CIF events, independent of predicted phone identity.

    Every firing closes a predicted segment. Drop the final closing event,
    which acts as the predicted utterance endpoint. No reference count is used.
    Fractional crossings assume uniform mass within each encoder frame cell.
    Grid crossings use the right endpoint of the crossing cell; no interpolation.
    """
    count = int(cif.lengths[0])
    positions = cif.fire_positions[0, :max(count-1, 0)].detach().cpu()
    return dict(text_free=(positions * (duration / frame_count)).tolist(),
                text_free_frame=(positions.ceil() * (duration / frame_count)).tolist())


@torch.no_grad()
def evaluate(model, rows, vocab, output=None, autocast_dtype=None):
    if not rows:
        raise ValueError('Cannot evaluate an empty split')
    training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals = Counter()
    counts = {mode: {tol: Counter() for tol in (10, 20, 30)} for mode in ('text_free', 'text_free_frame')}
    inverse = {i: phone for phone, i in vocab.items()}
    normalize = lambda seq: ['d' if p == 'dx' else p for p in seq if p not in ('sil', 'q')]
    records = []
    try:
        for row in rows:
            batch = collate([row], device)
            # No label tensor enters recognition; supervised validation loss is separate.
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = model.recognize(batch['input_values'], batch['attention_mask'])
                supervised = model(**batch)
            cif = prediction['cif']
            frame_count = int(prediction['frame_lengths'][0])
            duration = len(read_audio(row)) / 16000
            hypothesis = [inverse[int(i)] for i in prediction['tokens'][0, :int(cif.lengths[0])]]
            reference = [p for p, _, _ in row['references']['ottc_arpabet']]
            times = predicted_boundaries(cif, duration, frame_count)
            reference_times = [float(r[1]) for r in row['references']['ottc_arpabet'][1:]]
            for mode, boundaries in times.items():
                for tol in counts[mode]:
                    counts[mode][tol].update(_scorer.precision_recall_counts(boundaries, reference_times, tol / 1000))
            totals.update(utterances=1, errors=edits(reference, hypothesis), phones=len(reference),
                          normalized_errors=edits(normalize(reference), normalize(hypothesis)),
                          normalized_phones=len(normalize(reference)),
                          ce_sum=float(supervised['ce_sum']), quantity_sum=float(supervised['quantity_sum']),
                          ctc_sum=float(supervised['ctc_sum']), ottc_sum=float(supervised['ottc_sum']),
                          quantity_abs_error=abs(float(cif.quantity[0]) - len(reference)))
            records.append(dict(utt=row['utt'], reference=reference, hypothesis=hypothesis,
                reference_boundaries=reference_times, boundaries=times,
                raw_predicted_quantity=float(cif.quantity[0]), fired_tokens=int(cif.lengths[0]),
                frame_count=frame_count, duration=duration))
    finally:
        model.train(training)
    result = dict(utterances=totals['utterances'], raw_PER_percent=100 * totals['errors'] / totals['phones'],
        normalized_nosil_PER_percent=100 * totals['normalized_errors'] / totals['normalized_phones'],
        ce_per_phone=totals['ce_sum'] / totals['phones'],
        quantity_per_phone=totals['quantity_sum'] / totals['phones'],
        ctc_per_phone=totals['ctc_sum'] / totals['phones'],
        ottc_per_utterance=totals['ottc_sum'] / totals['utterances'],
        quantity_MAE=totals['quantity_abs_error'] / totals['utterances'],
        counts=dict(totals), audio_only=True, forced_alignment=False, rvalue_source_sha256=sha(SOURCE),
        protocol=dict(reference='frozen TIMIT40 occurrence starts [1:], including silence; utterance endpoints excluded',
            prediction='all CIF events except final closing event; repeated phone events retained; no reference count or phone input',
            clock='duration / valid_encoder_frames; fractional uniform within-cell crossing and right-cell grid reported separately',
            matching='frozen IF-MDD directional greedy, label_aware=False, lenient=False; pooled counts',
            supervised_loss='reference lengths used only for teacher-forced validation CE; never for segmentation/PER'))
    for mode, by_tol in counts.items():
        result[mode] = {str(tol): _scorer.finalize_counts(c['precision_matches'], c['recall_matches'],
            c['predicted_boundaries'], c['reference_boundaries']) for tol, c in by_tol.items()}
    if output is not None:
        output = Path(output)
        output.write_text(json.dumps(result, indent=2) + '\n')
        output.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return result
