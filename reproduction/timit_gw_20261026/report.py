"""Pilot development comparison. Never rank partial epochs or use test data."""
import json
from reproduction.timit_gw_20261026.common import HERE, ROOT, write
from reproduction.timit_gw_20261026.diagnostics import summarize


def main():
    baseline = ROOT/'reproduction/timit_seed_sweep_20261022/models/ag__seed43'
    vocab = json.loads((baseline/'vocab_used.json').read_text())
    records = []
    for name, output in [('historical_ag_seed43', baseline)] + [(p.name, p) for p in sorted((HERE/'models').glob('*')) if p.is_dir()]:
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


if __name__ == '__main__': main()
