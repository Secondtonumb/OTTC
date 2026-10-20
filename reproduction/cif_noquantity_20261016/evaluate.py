"""Batched audio-only CIF evaluation with original frozen boundary scoring."""
from collections import Counter
import json
from pathlib import Path
import time

import torch

from reproduction.segmentation_metrics import _scorer, SOURCE
from reproduction.cif_noquantity_20261016.data import collate, sha
from reproduction.cif_timit_20261016.evaluate import edits


def boundaries(cif, index, duration, frames):
    positions=cif.fire_positions[index,:max(int(cif.lengths[index])-1,0)].detach().cpu()
    return {'text_free':(positions*(duration/frames)).tolist(),
            'text_free_frame':(positions.ceil()*(duration/frames)).tolist()}


@torch.no_grad()
def evaluate(model, rows, vocab, output=None, autocast_dtype=None, batch_size=8):
    if not rows:raise ValueError('Empty evaluation split')
    was_training=model.training;model.eval();device=next(model.parameters()).device
    totals=Counter();counts={m:{t:Counter() for t in [10,20,30]} for m in ['text_free','text_free_frame']}
    inverse={i:p for p,i in vocab.items()}
    normalize=lambda seq:['d' if p=='dx' else p for p in seq if p not in ('sil','q')]
    records=[];start=time.monotonic()
    order=sorted(rows,key=lambda r:r['duration'])
    try:
        for offset in range(0,len(order),batch_size):
            group=order[offset:offset+batch_size];batch=collate(group,device)
            with torch.autocast(device.type,dtype=autocast_dtype,enabled=autocast_dtype is not None):
                prediction=model.recognize(batch['input_values'],batch['attention_mask'])
                supervised=model(**batch)
            for key in ['ce_sum','quantity_sum','ctc_sum','ottc_sum']:
                totals[key]+=float(supervised[key])
            cif=prediction['cif']
            for i,row in enumerate(group):
                frame_count=int(prediction['frame_lengths'][i]);duration=row['samples']/16000
                hypothesis=[inverse[int(t)] for t in prediction['tokens'][i,:int(cif.lengths[i])]]
                reference=[p for p,_,_ in row['references']['ottc_arpabet']]
                reference_times=[float(s) for _,s,_ in row['references']['ottc_arpabet'][1:] if 0 < s < duration]
                times=boundaries(cif,i,duration,frame_count)
                for mode,events in times.items():
                    for tolerance,c in counts[mode].items():
                        c.update(_scorer.precision_recall_counts(events,reference_times,tolerance/1000))
                totals.update(utterances=1,errors=edits(reference,hypothesis),phones=len(reference),
                    normalized_errors=edits(normalize(reference),normalize(hypothesis)),
                    normalized_phones=len(normalize(reference)),quantity_abs_error=abs(float(cif.quantity[i])-len(reference)))
                records.append(dict(utt=row['utt'],subset=row.get('subset'),reference=reference,hypothesis=hypothesis,
                    reference_boundaries=reference_times,boundaries=times,raw_predicted_quantity=float(cif.quantity[i]),
                    fired_tokens=int(cif.lengths[i]),frame_count=frame_count,duration=duration))
    finally:model.train(was_training)
    result=dict(utterances=totals['utterances'],raw_PER_percent=100*totals['errors']/totals['phones'],
        normalized_nosil_PER_percent=100*totals['normalized_errors']/max(totals['normalized_phones'],1),
        ce_per_phone=totals['ce_sum']/totals['phones'],quantity_per_phone=totals['quantity_sum']/totals['phones'],
        ctc_per_phone=totals['ctc_sum']/totals['phones'],ottc_per_utterance=totals['ottc_sum']/totals['utterances'],
        quantity_MAE=totals['quantity_abs_error']/totals['utterances'],counts=dict(totals),
        audio_only=True,forced_alignment=False,rvalue_source_sha256=sha(SOURCE),
        elapsed_seconds=time.monotonic()-start,evaluation_batch_size=batch_size,
        protocol=dict(reference='source phone occurrence starts [1:], including source silence; utterance endpoints excluded',
            prediction='CIF fires except final closing event; repeats retained; no phone labels/counts input to recognize',
            clock='duration/valid encoder frames; fractional and right-cell crossings separately',
            matching='frozen IF-MDD directional greedy, label_aware=False, lenient=False',
            loss='teacher-forced CIF count only for supervised loss; quantity logged diagnostically and weighted zero in training'))
    for mode,by_tol in counts.items():
        result[mode]={str(t):_scorer.finalize_counts(c['precision_matches'],c['recall_matches'],c['predicted_boundaries'],c['reference_boundaries']) for t,c in by_tol.items()}
    if output:
        output=Path(output);output.write_text(json.dumps(result,indent=2)+'\n')
        output.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return result
