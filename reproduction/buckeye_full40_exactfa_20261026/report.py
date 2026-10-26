"""Completed-only seed means and robust statistics, with original scores alongside."""
import csv
import json
from pathlib import Path
import statistics
from reproduction.buckeye_full40_exactfa_20261026.evaluate import write

HERE=Path(__file__).resolve().parent


def main():
    groups={}; pending=[]
    for cost in ('ag','ce','jsd_standard','jdiv'):
        for seed in (42,43,44):
            name=f'{cost}__seed{seed}'; root=HERE/'models'/name
            if not (root/'COMPLETE.json').exists(): pending.append(name); continue
            record=json.loads((root/'FINAL_SUMMARY.json').read_text())
            for candidate in ('best_dev','best3','dev_selected'):
                actual=record['selected'] if candidate=='dev_selected' else candidate
                for split,result in record['results'][actual].items():
                    fa=result['modes']['forced_alignment']; old=result['original_forced_alignment']; b=fa['boundaries']['20']
                    row=dict(seed=seed,MAE_ms=fa['boundary_MAE_ms'],old_MAE_ms=old['boundary_MAE_ms'],
                             precision=b['precision_percent'],recall=b['recall_percent'],F1=b['f1_percent'],R=b['r_value_percent'],
                             improved_search_paths=result['improved_search_paths'],utterances=result['utterances'])
                    row.update({k:result['endpoint_errors'][k] for k in ('median_ms','p90_ms','p99_ms','over_100ms_percent','over_500ms_percent')})
                    groups.setdefault((cost,candidate,split),[]).append(row)
    rows=[]
    for (cost,candidate,split),group in groups.items():
        row=dict(cost=cost,candidate=candidate,split=split,n=len(group),seeds=[x['seed'] for x in group])
        for key in group[0]:
            if key=='seed': continue
            vals=[x[key] for x in group]
            row[key]=statistics.mean(vals); row[key+'_std']=statistics.stdev(vals) if len(vals)>1 else None
        rows.append(row)
    write(HERE/'RESULTS.json',dict(complete=not pending,pending=pending,rows=rows,evaluation_uploads=False,
        robust_stats='median/P90/P99 are per-seed endpoint quantiles averaged across seeds; all data retained'))
    if rows:
        p=HERE/'COMPARISON.csv'; tmp=p.with_suffix('.csv.tmp')
        with tmp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        tmp.replace(p)
    if not pending: write(HERE/'COMPLETE.json',dict(complete=True,models=12,sets=96,evaluation_uploads=False))
    print(json.dumps(dict(completed=12-len(pending),pending=pending),ensure_ascii=False))


if __name__=='__main__': main()
