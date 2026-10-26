"""Keep TIMIT dev/test separate, aggregating only completed Libri-trained runs."""
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
        rows = dict(timit_dev=[], timit_test=[])
        pending = []
        for cost in COSTS:
            for seed in SEEDS:
                output = HERE / 'models' / f'{cost}__seed{seed}'
                paths = {split: output / f'benchmark_best_dev_{split}.json' for split in rows}
                if (output / 'BENCHMARK_COMPLETE.json').exists() and all(path.exists() for path in paths.values()):
                    for split, path in paths.items():
                        result = json.loads(path.read_text())
                        expected_count = 400 if split == 'timit_dev' else 192
                        assert result['utterances'] == expected_count
                        assert result['dataset']['name'] == split
                        assert not result['checkpoint_selection']['timit_used']
                        rows[split].append(row_from_metrics(cost, seed, output, result))
                else:
                    state = output / 'LATEST_STATE.json'
                    pending.append(dict(cost=cost, seed=seed,
                                        state=json.loads(state.read_text()) if state.exists() else None))
        report = dict(protocol='Libri100-only training/dev selection; FP32 best-Libri-dev checkpoint; TIMIT is evaluation only',
                      seeds=list(SEEDS), runs=rows, pending=pending,
                      summary={split: summarize(group, SEEDS) for split, group in rows.items()},
                      complete=all(len(group) == 12 for group in rows.values()))
        temporary = HERE / 'COMPARISON.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(HERE / 'COMPARISON.json')
        for split, group in rows.items():
            path = HERE / f'{split.upper()}_RESULTS.csv'
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=('cost', 'seed', *METRICS, 'source'))
                writer.writeheader()
                writer.writerows(group)
            temporary.replace(path)
            write_summary(HERE / f'{split.upper()}_THREE_SEED_SUMMARY.csv', report['summary'][split])
        print(f"Libri100 -> TIMIT: {len(rows['timit_test'])}/12 complete; {len(pending)} pending.", flush=True)


if __name__ == '__main__':
    main()
