"""Original scorer plus local fragmentation audit and FP32 epoch averages."""
import gc
import json
import torch
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate as original_evaluate
from reproduction.timit_gw_20261026.common import sha, write
from reproduction.timit_gw_20261026.diagnostics import summarize


def evaluate(config, model, rows, vocab, processor, topo, output, **kwargs):
    result = original_evaluate(config, model, rows, vocab, processor, topo, output, **kwargs)
    diagnostics = summarize(output.with_suffix('.predictions.jsonl'), vocab, config['frame_ms'])
    assert abs(diagnostics['counts']['PER_percent'] - result['modes']['text_free']['nosil_PER_percent']) < 1e-8
    result['fragmentation'] = diagnostics
    result['gw_training_options'] = config['gw_options']
    write(output, result)
    return result


def average(paths, output, config):
    if output.exists(): return
    accumulated, members = None, []
    for path in paths:
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['recipe'] == config['name']
        members.append(dict(path=str(path), epoch=saved['epoch'], step=saved['step'], sha256=sha(path)))
        assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
        if accumulated is None:
            accumulated = {k: v.clone() for k, v in saved['model'].items()}
        else:
            for k, v in saved['model'].items():
                if v.is_floating_point(): accumulated[k].add_(v)
                else: assert torch.equal(accumulated[k], v)
    for v in accumulated.values():
        if v.is_floating_point(): v.div_(len(paths))
    temporary = output.with_suffix('.tmp')
    torch.save(dict(model=accumulated, recipe=config['name'], members=members), temporary)
    temporary.replace(output)
    write(output.with_suffix('.json'), dict(members=members, weights_sha256=sha(output), dtype='float32'))


def final_benchmarks(config, model, rows, vocab, processor, topo, output, run):
    if (output/'BENCHMARK_COMPLETE.json').exists(): return
    epoch = int(round(json.loads((output/'TRAIN_COMPLETE.json').read_text())['epoch']))
    candidates = dict(best_dev=output/'best_dev.pt', final=output/'epoch_weights'/f'epoch-{epoch:03d}.pt')
    for count in (5, 10):
        if epoch >= count:
            path = output/f'final_{count}_average.pt'
            average([output/'epoch_weights'/f'epoch-{e:03d}.pt' for e in range(epoch-count+1, epoch+1)], path, config)
            candidates[f'final_{count}'] = path
    model.cuda()
    results = {}
    for split in ('dev', 'test'):
        for name, path in candidates.items():
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(saved['model']); del saved; gc.collect()
            target = output/f'benchmark_{name}_{split}.json'
            result = json.loads(target.read_text()) if target.exists() else evaluate(config, model, rows[split], vocab, processor, topo, target)
            results.setdefault(name, {})[split] = result
        if split == 'dev':
            selected = min(candidates, key=lambda n: results[n]['dev']['modes']['text_free']['nosil_PER_percent'])
            write(output/'DEV_SELECTION.json', dict(selected=selected, test_used=False, criterion='minimum development PER'))
    write(output/'BENCHMARK_COMPLETE.json', dict(selected=selected, test_used_for_selection=False,
        candidates={n:dict(path=str(p),sha256=sha(p)) for n,p in candidates.items()}))
