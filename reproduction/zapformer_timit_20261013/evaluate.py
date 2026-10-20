"""TIMIT40 recognition, text-free boundaries and supplied-phone FA, separately."""
import bootstrap
from collections import Counter
from itertools import groupby
import json

import torch
from data import Inputs
from segmentation_metrics import _scorer
from ifmdd_k2_alignment import align, load_backend


def edits(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


@torch.no_grad()
def evaluate(model, rows, vocab, *, forced=False, output=None):
    training = model.training
    model.eval()
    inputs = Inputs(rows)
    backend = load_backend()[0] if forced else None
    totals = Counter()
    counts = {mode: {tol: Counter() for tol in (10,20,30)} for mode in (('text_free','forced_k2') if forced else ('text_free',))}
    normalize = lambda seq: [vocab['d'] if p == vocab['dx'] else p for p in seq if p not in (vocab['sil'],vocab['q'])]
    records = []
    try:
        for row in rows:
            batch = inputs.batch([row['id']])
            feature = batch['inputs'].cuda()
            lengths = batch['supervisions']['num_frames'].cuda()
            logits, mass, mask = model.scores(feature, lengths)
            loss = model.criterion(logits, mass, mask, batch['tokens'].cuda(), batch['token_lengths'].cuda())
            frame_ids = logits[0].argmax(-1).cpu().tolist()
            hyp = [k for k,_ in groupby(frame_ids) if k != 0]
            ref = row['tokens']
            totals.update(errors=edits(ref,hyp), phones=len(ref), utterances=1, loss=float(loss),
                          nosil_errors=edits(normalize(ref),normalize(hyp)), nosil_phones=len(normalize(ref)))
            reference_times = [r[1] for r in row['references'][1:]]
            predicted = {'text_free': [i * row['duration']/len(frame_ids) for i in range(1,len(frame_ids)) if frame_ids[i] != frame_ids[i-1]]}
            if forced:
                segments,_ = align(logits[0].log_softmax(-1), ref, row['duration'], backend)
                predicted['forced_k2'] = [s['start'] for s in segments[1:]]
                totals['fa_abs_ms'] += sum(abs(a-b)*1000 for a,b in zip(predicted['forced_k2'],reference_times))
                totals['fa_boundaries'] += len(reference_times)
            for mode, times in predicted.items():
                for tol in counts[mode]:
                    counts[mode][tol].update(_scorer.precision_recall_counts(times, reference_times, tol/1000))
            records.append(dict(utt=row['id'], reference=ref, hypothesis=hyp, boundaries=predicted))
    finally:
        model.train(training)
    result = dict(utterances=totals['utterances'], raw_PER_percent=100*totals['errors']/totals['phones'],
                  normalized_nosil_PER_percent=100*totals['nosil_errors']/totals['nosil_phones'],
                  loss_per_utterance=totals['loss']/totals['utterances'], counts=dict(totals),
                  protocol='TIMIT40 raw PER; no-sil removes sil/q, maps dx->d. All internal token boundaries incl silence, endpoints excluded, duration/T, frozen directional matching. FA supplied phones; text_free argmax changes incl blank.')
    for mode, by_tol in counts.items():
        result[mode] = {str(tol): _scorer.finalize_counts(c['precision_matches'],c['recall_matches'],c['predicted_boundaries'],c['reference_boundaries']) for tol,c in by_tol.items()}
    if forced:
        result['forced_k2_mae_ms'] = totals['fa_abs_ms']/totals['fa_boundaries']
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        output.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return result
