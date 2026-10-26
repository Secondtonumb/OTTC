"""Local-only complete-run results for the twelve independent Buckeye recipes."""
import csv
import fcntl
import json
from pathlib import Path

from reproduction.timit_seed_sweep_20261022.report import row_from_metrics, summarize, write_summary, METRICS

HERE = Path(__file__).resolve().parent
COSTS = ('ce', 'jsd_standard', 'ag', 'jdiv')
SEEDS = (42, 43, 44)


def main():
    with (HERE / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, pending = [], []
        for cost in COSTS:
            for seed in SEEDS:
                output = HERE / 'models' / f'{cost}__seed{seed}'
                metrics = output / 'benchmark_best_dev_test.json'
                if (output / 'BENCHMARK_COMPLETE.json').exists() and metrics.exists():
                    rows.append(row_from_metrics(cost, seed, output, json.loads(metrics.read_text())))
                else:
                    state = output / 'LATEST_STATE.json'
                    pending.append(dict(cost=cost, seed=seed,
                                        state=json.loads(state.read_text()) if state.exists() else None))
        summaries = summarize(rows, SEEDS)
        report = dict(protocol='Buckeye human partial19, FP32 best-dev, no test selection; no mixing historical20ms/final models',
                      seeds=list(SEEDS), runs=rows, pending=pending, summary=summaries, complete=len(rows) == 12)
        temporary = HERE / 'COMPARISON.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(HERE / 'COMPARISON.json')
        temporary = HERE / 'COMPARISON.csv.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('cost', 'seed', *METRICS, 'source'))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(HERE / 'COMPARISON.csv')
        write_summary(HERE / 'THREE_SEED_SUMMARY.csv', summaries)
        print(f"Buckeye: {len(rows)}/12 complete; {len(pending)} pending.", flush=True)


if __name__ == '__main__':
    main()
