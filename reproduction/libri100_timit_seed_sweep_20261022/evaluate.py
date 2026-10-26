"""FP32 TIMIT-only evaluation of the single best Libri development checkpoint."""
import gc
import json

import torch

from reproduction.libri100_timit_seed_sweep_20261022.common import sha, write
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def final_benchmarks(config, model, rows, vocab, processor, topo, output, run):
    if (output / 'BENCHMARK_COMPLETE.json').exists():
        return
    assert not hasattr(model, '_original_forward')
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    path = output / 'best_dev.pt'
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(saved['model'])
    model.step = saved['step']
    del saved
    gc.collect()
    model.cuda()
    results = {}
    write(output / 'DEV_SELECTION.json', dict(selected='best_dev', test_used=False, timit_dev_used=False,
          selection_corpus=config['selection_corpus'], criterion=config['selection_reference']))
    for split, count in (('timit_dev', 400), ('timit_test', 192)):
        assert len(rows[split]) == count
        target = output / f'benchmark_best_dev_{split}.json'
        summary = json.loads(target.read_text()) if target.exists() else evaluate(
            config, model, rows[split], vocab, processor, topo, target)
        assert summary['utterances'] == count
        pfer = score_predictions(target.with_suffix('.predictions.jsonl'), summary)
        summary['modes']['text_free']['PFER_percent'] = pfer['PFER_percent']
        summary['pfer'] = pfer
        summary['dataset'] = dict(name=split, **config['expected_data'][split])
        summary['checkpoint_selection'] = dict(corpus=config['selection_corpus'], timit_used=False,
                                                checkpoint=str(path), step=model.step)
        write(target, summary)
        assert summary['modes']['forced_alignment']['alignment_failures'] == 0
        results[split] = summary
    write(output / 'FINAL_SUMMARY.json', dict(seed=config['seed'], cost=config['cost'],
          selected='best_dev', **results))
    write(output / 'BENCHMARK_COMPLETE.json', dict(selected='best_dev', test_used_for_selection=False,
          timit_dev_used_for_selection=False, candidates=dict(best_dev=dict(path=str(path), sha256=sha(path))),
          evaluated_splits=list(results)))
