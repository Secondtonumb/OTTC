"""Aggregate completed matched-seed results without treating queued jobs as results."""
import csv
import fcntl
import json
from pathlib import Path
import statistics
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'timit_ssl_native10_20261019/models'
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')
METRICS = ('PER', 'PFER', 'text_free_precision', 'text_free_recall', 'text_free_F1', 'text_free_R',
           'FA_precision', 'FA_recall', 'FA_F1', 'FA_R', 'FA_MAE_ms')


def summarize(rows, seeds):
    identities = [(row['cost'], row['seed']) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate objective/seed results')
    summaries = []
    for cost in COSTS:
        group = [row for row in rows if row['cost'] == cost and row['seed'] in seeds]
        record = dict(cost=cost, seeds=sorted(row['seed'] for row in group), n=len(group),
                      expected_seeds=list(seeds), complete=len(group) == len(seeds), metrics={})
        for key in METRICS:
            values = [row[key] for row in group if row[key] is not None]
            record['metrics'][key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                         sample_std=statistics.stdev(values) if len(values) > 1 else None)
        summaries.append(record)
    return summaries


def write_summary(path, summaries):
    fields = ['cost', 'seeds', 'n', 'complete'] + [f'{key}_{stat}' for key in METRICS for stat in ('mean', 'sample_std')]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in summaries:
            row = dict(cost=record['cost'], seeds=';'.join(map(str, record['seeds'])), n=record['n'], complete=record['complete'])
            row.update({f'{key}_{stat}': record['metrics'][key][stat] for key in METRICS for stat in ('mean', 'sample_std')})
            writer.writerow(row)
    temporary.replace(path)


def row_from_metrics(cost, seed, source, result):
    text = result['modes']['text_free']
    forced = result['modes']['forced_alignment']
    return dict(cost=cost, seed=seed, source=str(source), PER=text['nosil_PER_percent'],
                PFER=text.get('PFER_percent'), text_free_F1=text['boundaries']['20']['f1_percent'],
                text_free_precision=text['boundaries']['20']['precision_percent'],
                text_free_recall=text['boundaries']['20']['recall_percent'],
                text_free_R=text['boundaries']['20']['r_value_percent'],
                FA_F1=forced['boundaries']['20']['f1_percent'], FA_R=forced['boundaries']['20']['r_value_percent'],
                FA_precision=forced['boundaries']['20']['precision_percent'],
                FA_recall=forced['boundaries']['20']['recall_percent'],
                FA_MAE_ms=forced.get('boundary_MAE_ms'))


def main():
    lock = (HERE / '.report.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows, pending, historical = [], [], []
    for cost in COSTS:
        for seed in ([42, 43, 44, 45] if cost == 'jdiv' else [43, 44, 45]):
            output = HERE / 'models' / f'{cost}__seed{seed}'
            complete = output / 'BENCHMARK_COMPLETE.json'
            metrics = output / 'benchmark_best_dev_test.json'
            if complete.exists() and metrics.exists():
                rows.append(row_from_metrics(cost, seed, output, json.loads(metrics.read_text())))
            else:
                state = output / 'LATEST_STATE.json'
                pending.append(dict(cost=cost, seed=seed, status='not_yet_benchmarked',
                                    state=json.loads(state.read_text()) if state.exists() else None))
    for cost in COSTS[:-1]:
        source = BASE / f"wav2vec2__timit__10ms__mono__{'jsd_upstream' if cost == 'ag' else cost}"
        metric = source / 'benchmark_best_dev_test.json'
        if metric.exists():
            result = json.loads(metric.read_text())
            sidecar = HERE / f'historical_{cost}_seed42_pfer.json'
            if sidecar.exists():
                pfer = json.loads(sidecar.read_text())
            else:
                pfer = score_predictions(metric.with_suffix('.predictions.jsonl'), result)
                sidecar.write_text(json.dumps(pfer, indent=2) + '\n')
            result['modes']['text_free']['PFER_percent'] = pfer['PFER_percent']
            historical.append(row_from_metrics(cost, 42, source, result))
    summaries = summarize(rows, (43, 44, 45))
    for record in summaries:
        record.update(completed_new_seeds=record['n'], expected_new_seeds=3)
    matched_seeds = (42, 43, 44)
    matched = summarize(historical + rows, matched_seeds)
    all_seeds = summarize(historical + rows, (42, 43, 44, 45))
    report = dict(protocol='FP32 single best-dev checkpoint, same data/scorer; no checkpoint-average mixing',
                  new_runs=rows, pending=pending, historical_seed42=historical, three_seed_summary=summaries,
                  matched_seeds=list(matched_seeds), matched_seed_summary=matched,
                  matched_seed_complete=all(record['complete'] for record in matched),
                  all_seed_summary=all_seeds,
                  complete=len(rows) == 13)
    temporary = HERE / 'COMPARISON.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(HERE / 'COMPARISON.json')
    with (HERE / 'COMPARISON.csv').open('w', newline='') as stream:
        fields = ('cost', 'seed', 'PER', 'PFER', 'text_free_precision', 'text_free_recall', 'text_free_F1', 'text_free_R',
                  'FA_precision', 'FA_recall', 'FA_F1', 'FA_R', 'FA_MAE_ms', 'source')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(historical + rows)
    write_summary(HERE / 'MERGED_COMMON_SEEDS.csv', matched)
    write_summary(HERE / 'MERGED_ALL_SEEDS.csv', all_seeds)
    print(f'Comparison updated: {len(rows)}/13 new runs benchmarked; {len(pending)} pending.', flush=True)


if __name__ == '__main__':
    main()
