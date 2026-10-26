"""Local report, keeping incomplete seeds and checkpoint averages explicit."""
import csv
import fcntl
import json
import statistics
from reproduction.timit_frame_ce_20261026.common import HERE, ROOT, write


def metrics(result):
    tf = result['modes']['text_free']
    fa = result['modes']['forced_alignment']
    values = dict(PER=tf['nosil_PER_percent'], PFER=tf['PFER_percent'], FA_MAE_ms=fa['boundary_MAE_ms'])
    for prefix, mode in (('text_free', tf), ('FA', fa)):
        for name, key in (('precision', 'precision_percent'), ('recall', 'recall_percent'), ('F1', 'f1_percent'), ('R', 'r_value_percent')):
            values[f'{prefix}_{name}'] = mode['boundaries']['20'][key]
    return values


def main():
    lock = (HERE / '.report.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    entries = []
    statuses = {}
    for seed in (42, 43, 44):
        directory = HERE / 'models' / f'frame_ce__seed{seed}'
        done = directory / 'BENCHMARK_COMPLETE.json'
        latest = directory / 'LATEST_STATE.json'
        statuses[str(seed)] = dict(benchmark_complete=done.exists(), training_complete=(directory / 'TRAIN_COMPLETE.json').exists(),
                                  latest=json.loads(latest.read_text()) if latest.exists() else None)
        if not done.exists():
            continue
        for candidate in ('best_dev', 'final_5', 'final_10'):
            for split in ('dev', 'test'):
                target = directory / f'benchmark_{candidate}_{split}.json'
                result = json.loads(target.read_text())
                assert result['utterances'] == (400 if split == 'dev' else 192)
                entries.append(dict(cost='frame_ce', seed=seed, candidate=candidate, split=split, source=str(target), **metrics(result)))
    primary = [r for r in entries if r['candidate'] == 'best_dev' and r['split'] == 'test']
    keys = [k for k in primary[0] if isinstance(primary[0][k], (int, float)) and k != 'seed'] if primary else []
    aggregate = dict(n=len(primary), seeds=[r['seed'] for r in primary], complete=len(primary) == 3)
    if primary:
        aggregate['metrics'] = {k: dict(mean=statistics.mean(r[k] for r in primary),
            sample_std=statistics.stdev(r[k] for r in primary) if len(primary) > 1 else None) for k in keys}
    historical = ROOT / 'reproduction/timit_seed_sweep_20261022/COMPARISON.json'
    previous = json.loads(historical.read_text())
    write(HERE / 'COMPARISON.json', dict(protocol='primary FP32 single best-dev PER; same manifests and scorer; averages separate; frame CE uses gold training times',
        status=statuses, frame_ce_primary_test=aggregate, results=entries,
        historical_source=str(historical), historical=previous))
    if entries:
        with (HERE / 'RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader(); writer.writerows(entries)
    lines = ['# TIMIT Frame-wise CE', '', 'Gold PHN frame-supervised baseline; compare primary single best-dev checkpoints with the original sweep.', '',
             '| Seed | Training complete | Benchmark complete |', '|---|---|---|']
    lines += [f"| {s} | {v['training_complete']} | {v['benchmark_complete']} |" for s, v in statuses.items()]
    lines += ['', f"Completed primary test seeds: {aggregate['n']}/3. Incomplete seeds are not assigned scores.", '',
              'Detailed dev/test recognition, text-free/FA boundaries, and separate final5/final10 results: COMPARISON.json and RESULTS.csv.']
    lines += ['', 'Single best-dev checkpoint, test; mean ± sample standard deviation across seeds42–44.', '',
              '| Method | n | PER % | PFER % | Text-free F1/R % | FA F1/R % | FA MAE ms |',
              '|---|---:|---:|---:|---|---|---:|']
    summaries = previous['matched_seed_summary'] + [dict(cost='frame_ce (gold frame supervision)', **aggregate)]
    for summary in summaries:
        def fmt(key):
            item = summary.get('metrics', {}).get(key)
            if not item:
                return 'pending'
            return f"{item['mean']:.2f} ± {item['sample_std']:.2f}" if item['sample_std'] is not None else f"{item['mean']:.2f} (one seed)"
        lines.append(f"| {summary['cost']} | {summary['n']} | {fmt('PER')} | {fmt('PFER')} | {fmt('text_free_F1')} / {fmt('text_free_R')} | {fmt('FA_F1')} / {fmt('FA_R')} | {fmt('FA_MAE_ms')} |")
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
