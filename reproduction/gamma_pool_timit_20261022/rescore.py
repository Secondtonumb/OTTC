"""Regenerate native SpeechBrain reports from saved predictions; no inference."""
import argparse
import csv
import inspect
import json
from pathlib import Path
import time

from reproduction.gamma_pool_timit_20261022.evaluate import (
    HERE, MODES, score, report, sha, write,
)
from speechbrain.dataio import wer
from speechbrain.utils import edit_distance, metric_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    output = args.output
    complete = json.loads((output/'COMPLETE.json').read_text())
    assert complete['complete'] and complete['full_sets']
    original = json.loads((output/'RESULTS.json').read_text())
    results = {}
    changes = []
    for cost in complete['costs']:
        results[cost] = {}
        for split, expected in [('dev', 400), ('test', 192)]:
            records = [json.loads(line) for line in (output/f'{cost}_{split}.predictions.jsonl').open()]
            assert len(records) == len({r['utt'] for r in records}) == expected
            results[cost][split] = {}
            for mode in records[0]['hypotheses']:
                totals = score([r['reference'] for r in records], [r['hypotheses'][mode] for r in records])
                totals['reference_length_used'] = mode == 'gamma_reference_length'
                # Tie breaking may redistribute S/D/I; total edits must agree.
                assert totals['errors'] == original[cost][split][mode]['errors']
                assert totals['reference_phones'] == original[cost][split][mode]['reference_phones']
                changes.append(dict(cost=cost, split=split, mode=mode,
                    before=original[cost][split][mode], after=totals))
                results[cost][split][mode] = totals
                report(output/f'{cost}_{split}_{mode}.wer.txt', records, mode, totals)
    write(output/'RESULTS.json', results)
    with (output/'RESULTS.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cost', 'split', 'decoder', *totals])
        writer.writeheader()
        for cost, splits in results.items():
            for split, modes in splits.items():
                for mode, values in modes.items():
                    writer.writerow(dict(cost=cost, split=split, decoder=mode, **values))
    write(output/'SPEECHBRAIN_REPORT_PROVENANCE.json', dict(
        scorer='native speechbrain.utils.metric_stats.ErrorRateStats.write_stats',
        source_sha256={str(p):sha(p) for p in [Path(__file__), HERE/'evaluate.py',
            Path(inspect.getfile(wer)), Path(inspect.getfile(edit_distance)), Path(inspect.getfile(metric_stats))]},
        inference_rerun=False, predictions_unchanged=True, changes=changes,
        time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    complete.update(results_sha256=sha(output/'RESULTS.json'),
        scorer='SpeechBrain ErrorRateStats', reports_column_aligned=True,
        report_provenance_sha256=sha(output/'SPEECHBRAIN_REPORT_PROVENANCE.json'))
    write(output/'COMPLETE.json', complete)


if __name__ == '__main__':
    main()
