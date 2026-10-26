"""Report only completed frame-supervised benchmarks; keep averages separate."""
import csv
import fcntl
import json
import statistics
from reproduction.timit_frame_divergences_20261026.common import HERE, ROOT, COSTS, SEEDS, write
from reproduction.timit_frame_ce_20261026.report import metrics


def main():
    lock = (HERE / '.report.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    entries, statuses, summaries = [], {}, []
    families = [('frame_ce', ROOT / 'reproduction/timit_frame_ce_20261026')]
    families += [(f'frame_{cost}', HERE) for cost in COSTS]
    for method, root in families:
        primary = []
        for seed in SEEDS:
            name = f'{method}__seed{seed}'
            directory = root / 'models' / name
            done = directory / 'BENCHMARK_COMPLETE.json'
            history = directory / 'history.jsonl'
            latest = json.loads(history.read_text().splitlines()[-1]) if history.exists() else None
            statuses[name] = dict(training_complete=(directory / 'TRAIN_COMPLETE.json').exists(),
                                  benchmark_complete=done.exists(), latest=latest)
            if not done.exists():
                continue
            for candidate in ('best_dev', 'final_5', 'final_10'):
                for split in ('dev', 'test'):
                    target = directory / f'benchmark_{candidate}_{split}.json'
                    result = json.loads(target.read_text())
                    assert result['utterances'] == (400 if split == 'dev' else 192)
                    row = dict(method=method, seed=seed, candidate=candidate, split=split, source=str(target), **metrics(result))
                    entries.append(row)
                    if candidate == 'best_dev' and split == 'test':
                        primary.append(row)
        summary = dict(method=method, n=len(primary), seeds=[r['seed'] for r in primary], complete=len(primary) == len(SEEDS))
        if primary:
            summary['metrics'] = {k: dict(mean=statistics.mean(r[k] for r in primary),
                sample_std=statistics.stdev(r[k] for r in primary) if len(primary)>1 else None) for k in metrics(result)}
        summaries.append(summary)
    write(HERE / 'COMPARISON.json', dict(protocol='gold-frame supervised; FP32 single best-dev checkpoint primary; averages separate; dev-only selection',
        status=statuses, primary_test_summary=summaries, results=entries,
        original_OTTC_sweep_source=str(ROOT / 'reproduction/timit_seed_sweep_20261022/COMPARISON.json')))
    if entries:
        with (HERE / 'RESULTS.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0])); writer.writeheader(); writer.writerows(entries)
    lines = ['# TIMIT gold-frame CE / JSD / J-DIV / AG', '',
        'Primary: FP32 single best-dev checkpoint; seeds42/43/44. Missing seeds remain pending. Averages are separate in COMPARISON.json/RESULTS.csv.', '',
        '| Method | Completed test seeds | PER % | PFER % | Text-free F1/R % | FA F1/R % | FA MAE ms |',
        '|---|---:|---:|---:|---|---|---:|']
    for summary in summaries:
        def fmt(key):
            value = summary.get('metrics', {}).get(key)
            if not value:
                return 'pending'
            suffix = f" ± {value['sample_std']:.2f}" if value['sample_std'] is not None else ' (one seed)'
            return f"{value['mean']:.2f}" + suffix
        lines.append(f"| {summary['method']} | {summary['n']}/3 | {fmt('PER')} | {fmt('PFER')} | {fmt('text_free_F1')} / {fmt('text_free_R')} | {fmt('FA_F1')} / {fmt('FA_R')} | {fmt('FA_MAE_ms')} |")
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
