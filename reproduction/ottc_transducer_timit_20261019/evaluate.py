"""Free recognition and mass-quantile segmentation, with frozen IF-MDD scoring."""
from collections import Counter
import json
from pathlib import Path

import torch

from .data import collate, read_audio, sha
def edits(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


from . import scorer as _scorer
SOURCE = Path(__file__).with_name("scorer.py")


@torch.no_grad()
def evaluate(model, rows, vocab, output=None, autocast_dtype=None):
    if not rows:
        raise ValueError('empty evaluation split')
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts, records = Counter(), {t:Counter() for t in (10,20,30)}, []
    inverse = {v:k for k,v in vocab.items()}
    normalize = lambda seq:['d' if p=='dx' else p for p in seq if p not in ('sil','q')]
    try:
        for row in rows:
            batch = collate([row], device)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = model.recognize(batch['input_values'], batch['attention_mask'])
                supervised = model(**batch)
            hyp = [inverse[i] for i in prediction['tokens'][0]]
            reference = [p for p,_,_ in row['references']['ottc_arpabet']]
            duration, frames = len(read_audio(row))/16000, int(prediction['frame_lengths'][0])
            times = (prediction['boundary_positions'][0]*duration/frames).cpu().tolist()
            ref_times = [float(x[1]) for x in row['references']['ottc_arpabet'][1:] if 0<float(x[1])<duration]
            errors = edits(reference,hyp)
            for tolerance,counter in counts.items():
                counter.update(_scorer.precision_recall_counts(times,ref_times,tolerance/1000))
            repeated = sum(a==b for a,b in zip(reference,reference[1:]))
            totals.update(utterances=1, errors=errors, phones=len(reference), predicted_phones=len(hyp),
                normalized_errors=edits(normalize(reference),normalize(hyp)), normalized_phones=len(normalize(reference)),
                ottc_sum=float(supervised['ottc_sum']), length_sum=float(supervised['length_sum']),
                count_abs_error=abs(len(hyp)-len(reference)), length_clipped=int(prediction['length_clipped'][0]),
                repeated_reference_pairs=repeated)
            if repeated:
                totals.update(repeated_target_utterances=1, repeated_target_errors=errors,
                    repeated_target_phones=len(reference), repeated_target_count_abs_error=abs(len(hyp)-len(reference)))
            records.append(dict(utt=row['utt'], reference=reference, hypothesis=hyp,
                reference_boundaries=ref_times, text_free_boundaries=times, predicted_count=len(hyp),
                predicted_log_count=float(prediction['predicted_log_count'][0]),
                reference_count=len(reference), length_clipped=bool(prediction['length_clipped'][0]),
                repeated_reference_pairs=repeated, frame_count=frames, duration=duration))
    finally:
        model.train(was_training)
    result = dict(utterances=totals['utterances'], raw_PER_percent=100*totals['errors']/totals['phones'],
        normalized_nosil_PER_percent=100*totals['normalized_errors']/max(1,totals['normalized_phones']),
        ottc_per_utterance=totals['ottc_sum']/totals['utterances'],
        length_log_huber_per_utterance=totals['length_sum']/totals['utterances'],
        predicted_count_MAE=totals['count_abs_error']/totals['utterances'],
        repeated_target_PER_percent=100*totals['repeated_target_errors']/totals['repeated_target_phones']
            if totals['repeated_target_phones'] else None,
        counts=dict(totals), audio_only=True, forced_alignment=False, blank_free=True,
        rvalue_source_sha256=sha(SOURCE),
        protocol=dict(prediction='internal quantiles k/predicted_count of learned temporal mass; fractional frame cells; no target count, no repeat collapse',
            clock='boundary position * audio duration / valid encoder frames; utterance endpoints excluded',
            matching='frozen IF-MDD directional greedy pooled P/R/F1/R at 10/20/30ms; label_aware=False, lenient=False',
            reference='frozen TIMIT40 occurrence starts [1:]; utterance endpoints excluded',
            interpretation='count-conditioned OT mass segmentation; model predicts count independently from audio',
            supervised_loss='reference labels/count are used only for validation loss; recognize() takes only audio/mask'))
    result['objective_per_utterance'] = result['ottc_per_utterance']+model.phone_head.length_weight*result['length_log_huber_per_utterance']
    result['text_free'] = {str(t):_scorer.finalize_counts(c['precision_matches'],c['recall_matches'],
        c['predicted_boundaries'],c['reference_boundaries']) for t,c in counts.items()}
    if output is not None:
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
        Path(output).with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    return result


def flat_metrics(prefix,result):
    out = {f'{prefix}/{k}':result[k] for k in ('raw_PER_percent','normalized_nosil_PER_percent',
        'objective_per_utterance','ottc_per_utterance','length_log_huber_per_utterance','predicted_count_MAE','repeated_target_PER_percent')}
    out[f'{prefix}/length_clipped'] = result['counts'].get('length_clipped',0)
    for tolerance,metric in result['text_free'].items():
        for key in ('precision_pct','recall_pct','f1_pct','r_value_pct'):
            out[f'{prefix}/text_free/{key}_{tolerance}ms'] = metric[key]
    return out
