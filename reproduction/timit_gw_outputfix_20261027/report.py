"""Pilot development comparison. Never rank partial epochs or use test data."""
import json
from reproduction.timit_gw_outputfix_20261027.common import HERE, ROOT, write
from reproduction.timit_gw_outputfix_20261027.diagnostics import summarize


def main():
    baseline = ROOT/'reproduction/timit_seed_sweep_20261022/models/ag__seed43'
    vocab = json.loads((baseline/'vocab_used.json').read_text())
    records = []
    planned = ['monotone_w0.1__seed43', 'asot_w0.1__seed43', 'asot_w0__seed43']
    for name, output in [('historical_ag_seed43', baseline)] + [(name, (ROOT/'reproduction/timit_gw_20261026' if name.startswith('monotone') else HERE)/'models'/name) for name in planned]:
        target = output/'development/epoch-005.json'
        if not target.exists():
            records.append(dict(name=name, status='no completed epoch5 development artifact'))
            continue
        result = json.loads(target.read_text())
        fragment = result.get('fragmentation') or summarize(target.with_suffix('.predictions.jsonl'), vocab)
        records.append(dict(name=name, status='epoch5 development evaluated', epoch=5,
            PER=result['modes']['text_free']['nosil_PER_percent'],
            boundaries=result['modes']['text_free']['boundaries'],
            fragmentation=fragment['counts'], evidence=str(target)))
    write(HERE/'PILOT_COMPARISON.json', dict(records=records, test_used=False,
        warning='single seed pilot within 500-step warmup; cannot establish final recognition gains; historical baseline is not rerun'))
    full = []
    sources = [('historical_ag_seed43', baseline)] + [(name,
        (ROOT/'reproduction/timit_gw_20261026' if name.startswith('monotone') else HERE)/'models'/name) for name in planned]
    for name, output in sources:
        complete = output/'BENCHMARK_COMPLETE.json'
        if not complete.exists():
            latest = output/'LATEST_STATE.json'
            full.append(dict(name=name, status='full benchmark incomplete',
                             latest=json.loads(latest.read_text()) if latest.exists() else None))
            continue
        selected = json.loads((output/'DEV_SELECTION.json').read_text())['selected']
        candidates = {}
        for candidate in sorted(set(['best_dev', selected])):
            splits = {}
            for split in ('dev','test'):
                target = output/f'benchmark_{candidate}_{split}.json'
                metrics = json.loads(target.read_text())
                fragment = metrics.get('fragmentation') or summarize(target.with_suffix('.predictions.jsonl'), vocab)
                assert abs(fragment['counts']['PER_percent'] - metrics['modes']['text_free']['nosil_PER_percent']) < 1e-8
                splits[split] = dict(modes=metrics['modes'], fragmentation=fragment['counts'], evidence=str(target))
            candidates[candidate] = splits
        full.append(dict(name=name,status='full benchmark complete',selected_using_development=selected,candidates=candidates))
    write(HERE/'FULL_COMPARISON.json',dict(records=full,
        comparison='best_dev vs best_dev; dev-selected averages separately; seed43 only; completed historical AG baseline reused',
        test_used_for_selection=False))


if __name__ == '__main__': main()
