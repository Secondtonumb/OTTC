"""Local best-dev versus final-three average; choose on dev before test."""
import gc
import json
from pathlib import Path

import torch

from reproduction.timit_divergences_20261022.common import sha, write
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate


def average_checkpoints(paths, output):
    """Average model weights, retaining member epochs/steps/content hashes."""
    paths = [Path(p) for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Provide distinct checkpoints to average')
    result = None
    members = []
    identity = None
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        current = {k: saved[k] for k in ('recipe', 'config_sha256', 'source_sha256')}
        if identity is None:
            identity = current
        if identity != current:
            raise ValueError('Average members have different training identities')
        state = saved['model']
        if result is None:
            result = {k: v.clone() if not v.is_floating_point() else
                      v.to(torch.float64 if v.dtype == torch.float64 else torch.float32).clone()
                      for k, v in state.items()}
        else:
            if result.keys() != state.keys():
                raise ValueError('Average members have different model keys')
            for k, v in state.items():
                if result[k].shape != v.shape:
                    raise ValueError(f'Average shape mismatch: {k}')
                if v.is_floating_point():
                    result[k].add_(v)
                elif not torch.equal(result[k], v):
                    raise ValueError(f'Non-floating buffer differs: {k}')
        members.append(dict(path=str(path.resolve()), epoch=saved['epoch'],
                            step=saved['step'], sha256=sha(path)))
        del saved, state
    for value in result.values():
        if value.is_floating_point():
            value.div_(len(paths))
    payload = dict(model=result, **identity, members=members,
                   averaging='arithmetic mean of model weights only',
                   epoch=members[-1]['epoch'], step=members[-1]['step'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    write(output.with_suffix('.json'), dict(path=str(output), sha256=sha(output), members=members,
                                           averaging=payload['averaging']))
    return output


def final_benchmarks(config, model, rows, vocab, processor, topo, output, run):
    if (output/'BENCHMARK_COMPLETE.json').exists():
        return
    members = [output/'epoch_weights'/f'epoch-{e:03d}.pt' for e in config['average_epochs']]
    average = output/'averaged_weights'/'last3.pt'
    if not average.exists():
        average_checkpoints(members, average)
    ranked = json.loads((output/'BEST_THREE.json').read_text())['checkpoints']
    candidates = dict(best_dev=Path(ranked[0]['path']), avg_last3=average)
    mode = 'text_free'
    results = {}
    model.cuda()
    # All candidates' dev metrics are committed before opening any test rows.
    for split in ('dev', 'test'):
        for name, path in candidates.items():
            saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            model.load_state_dict(saved['model'])
            model.step = saved['step']
            del saved
            gc.collect()
            target = output/f'benchmark_{name}_{split}.json'
            result = json.loads(target.read_text()) if target.exists() else evaluate(
                config, model, rows[split], vocab, processor, topo, target)
            results.setdefault(name, {})[split] = result
            assert result['modes']['forced_alignment']['alignment_failures'] == 0, (name, split)
        if split == 'dev':
            selected = min(candidates, key=lambda n: results[n]['dev']['modes'][mode]['nosil_PER_percent'])
            write(output/'DEV_SELECTION.json', dict(selected=selected, test_used=False,
                criterion='minimum FP32 dev no-sil PER: best retained checkpoint vs final-three mean',
                candidates={n: str(p) for n, p in candidates.items()}))
    write(output/'BENCHMARK_COMPLETE.json', dict(selected=selected,
        test_used_for_selection=False, average_epochs=config['average_epochs'],
        candidates={n: dict(path=str(p), sha256=sha(p)) for n, p in candidates.items()}))
