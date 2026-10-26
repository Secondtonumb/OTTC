"""Historical evaluator, matched single best-dev and final checkpoints only."""
import gc
import json

import torch

from reproduction.timit_seed_sweep_20261022.common import sha, write
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def final_benchmarks(config, model, rows, vocab, processor, topo, output, run):
    if (output / 'BENCHMARK_COMPLETE.json').exists():
        return
    candidates = dict(best_dev=output / 'best_dev.pt')
    results = {}
    model.cuda()
    for split in ('dev', 'test'):
        for name, path in candidates.items():
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(saved['model'])
            model.step = saved['step']
            del saved
            gc.collect()
            target = output / f'benchmark_{name}_{split}.json'
            summary = json.loads(target.read_text()) if target.exists() else evaluate(
                config, model, rows[split], vocab, processor, topo, target)
            pfer = score_predictions(target.with_suffix('.predictions.jsonl'), summary)
            summary['modes']['text_free']['PFER_percent'] = pfer['PFER_percent']
            summary['pfer'] = pfer
            write(target, summary)
            assert summary['modes']['forced_alignment']['alignment_failures'] == 0
            results.setdefault(name, {})[split] = summary
        if split == 'dev':
            selected = min(candidates, key=lambda name: results[name]['dev']['modes']['text_free']['nosil_PER_percent'])
            write(output / 'DEV_SELECTION.json', dict(selected=selected, test_used=False,
                  criterion='single best development-PER checkpoint; no averages',
                  historical_comparison_candidate='best_dev'))
    write(output / 'BENCHMARK_COMPLETE.json', dict(selected=selected, test_used_for_selection=False,
          historical_comparison_candidate='best_dev', candidates={name: dict(path=str(path), sha256=sha(path))
                                                                 for name, path in candidates.items()}))
    write(output / 'FINAL_SUMMARY.json', dict(seed=config['seed'], cost=config['cost'],
          comparison_candidate='best_dev', dev=results['best_dev']['dev'], test=results['best_dev']['test']))
