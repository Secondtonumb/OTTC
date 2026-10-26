"""Matched per-seed GW deltas; aggregate only complete three-seed groups."""
import fcntl
import hashlib
import json
import statistics
from pathlib import Path
from reproduction.timit_monotone_gw_sweep_20261027.common import HERE,ROOT,COSTS,SEEDS,REUSED_AG43,sha,write
from reproduction.timit_monotone_gw_sweep_20261027.diagnostics import summarize
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def metrics(output,candidate,split):
    path=output/f'benchmark_{candidate}_{split}.json'
    pred=path.with_suffix('.predictions.jsonl')
    if not path.exists() or not pred.exists():return None
    d=json.loads(path.read_text())
    rows=[json.loads(line) for line in pred.open()]
    expected=400 if split=='dev' else 192
    assert d['utterances']==len(rows)==expected
    assert len({row['utt'] for row in rows})==expected
    reference_hash=hashlib.sha256(json.dumps(sorted((r['utt'],r['reference']) for r in rows),sort_keys=True).encode()).hexdigest()
    vocab=json.loads((output/'vocab_used.json').read_text())
    fragment=d.get('fragmentation') or summarize(pred,vocab)
    t=d['modes']['text_free'];f=d['modes']['forced_alignment']
    assert abs(fragment['counts']['PER_percent']-t['nosil_PER_percent'])<1e-8
    pfer=d.get('pfer')
    if pfer is None:
        cache=HERE/'derived_pfer'/f'{hashlib.sha256(str(path).encode()).hexdigest()}.json'
        signature=dict(metrics_sha256=sha(path),predictions_sha256=sha(pred))
        saved=json.loads(cache.read_text()) if cache.exists() else {}
        if saved.get('signature')!=signature:
            saved=dict(signature=signature,pfer=score_predictions(pred,d));write(cache,saved)
        pfer=saved['pfer']
    c=fragment['counts'];b=t['boundaries']['20']
    return dict(PER=t['nosil_PER_percent'],PFER=pfer['PFER_percent'],
        precision=b['precision_percent'],recall=b['recall_percent'],F1=b['f1_percent'],R=b['r_value_percent'],
        FA_R=f['boundaries']['20']['r_value_percent'],FA_MAE_ms=f['boundary_MAE_ms'],
        insertions=c['I'],deletions=c['D'],short_ABA=c['aba_middle_le20ms'],
        short_runs=c['predicted_runs_le20ms'],short_reference_deletion_rate=c['short_reference_deletion_rate'],
        evidence=str(path),reference_sha256=reference_hash)


def main():
    lock=(HERE/'.report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    records=[]
    for cost in COSTS:
        for seed in SEEDS:
            name=f'{cost}__seed{seed}'
            baseline=ROOT/'reproduction/timit_seed_sweep_20261022/models'/name
            output=REUSED_AG43 if name=='ag__seed43' else HERE/'models'/name
            record=dict(cost=cost,seed=seed,gw_path=str(output),reused=name=='ag__seed43',complete=False)
            record['baseline']={split:metrics(baseline,'best_dev',split) for split in ('dev','test')}
            if (output/'TRAIN_COMPLETE.json').exists() and (output/'BENCHMARK_COMPLETE.json').exists():
                complete=json.loads((output/'TRAIN_COMPLETE.json').read_text())
                assert complete['epoch']>=40
                record['best_dev']={split:metrics(output,'best_dev',split) for split in ('dev','test')}
                selection=json.loads((output/'DEV_SELECTION.json').read_text())['selected']
                record['dev_selected_candidate']=selection
                record['dev_selected']={split:metrics(output,selection,split) for split in ('dev','test')}
                record['complete']=all(record[group][split] is not None for group in ('baseline','best_dev','dev_selected') for split in ('dev','test'))
                if record['complete']:
                    for split in ('dev','test'):
                        assert record['baseline'][split]['reference_sha256']==record['best_dev'][split]['reference_sha256']==record['dev_selected'][split]['reference_sha256']
                    record['best_dev_minus_baseline']={split:{k:record['best_dev'][split][k]-record['baseline'][split][k]
                        for k in ('PER','PFER','F1','R','insertions','deletions','short_ABA','short_reference_deletion_rate')} for split in ('dev','test')}
            records.append(record)
    aggregates=[]
    for cost in COSTS:
        members=[r for r in records if r['cost']==cost]
        done=sum(r['complete'] for r in members)
        group=dict(cost=cost,completed=done,required=3,statistics=None)
        if done==3:
            group['statistics']={mode:{split:{k:dict(mean=statistics.mean(r[mode][split][k] for r in members),
                sample_sd=statistics.stdev(r[mode][split][k] for r in members)) for k in ('PER','PFER','F1','R','insertions','deletions','short_ABA','short_reference_deletion_rate')}
                for split in ('dev','test')} for mode in ('baseline','best_dev','dev_selected','best_dev_minus_baseline')}
        aggregates.append(group)
    write(HERE/'RESULTS.json',dict(records=records,aggregates=aggregates,
        complete=all(r['complete'] for r in records),test_used_for_selection=False,
        protocol='best-dev vs matched best-dev, selected averages separate; only all-three-complete groups have mean/sample SD; common references verified'))
    print('Completed',sum(r['complete'] for r in records),'/12',flush=True)


if __name__=='__main__':main()
