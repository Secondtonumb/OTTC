"""Local-only FP32 checkpoint comparison; select on Buckeye dev before tests."""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2Processor

from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def candidate_members(members):
    assert len(members) == 3 and len({item['epoch'] for item in members}) == 3
    ranked = sorted(members, key=lambda item: (item['dev_PER'], item['epoch']))
    return dict(best_dev=ranked[:1], best3=ranked)


def average(members, config, target):
    accumulated = None
    identity = None
    for member in members:
        path = Path(member['path'])
        assert common.base.sha(path) == member['sha256']
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['recipe'] == config['name'] and saved['seed'] == config['seed']
        assert saved['epoch'] == member['epoch'] and saved['step'] == member['step']
        assert saved['source_sha256'] == config['source_sha256']
        if identity is None:
            identity = saved['model_config']
        assert saved['model_config'] == identity
        state = saved['model']
        if accumulated is not None:
            assert set(state) == set(accumulated)
        for name, tensor in state.items():
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            if accumulated is not None:
                assert tensor.shape == accumulated[name].shape and tensor.dtype == accumulated[name].dtype
                if tensor.is_floating_point():
                    accumulated[name].add_(tensor)
                else:
                    assert torch.equal(accumulated[name], tensor)
        if accumulated is None:
            accumulated = {name: tensor.clone() for name, tensor in state.items()}
        del state, saved
    for tensor in accumulated.values():
        if tensor.is_floating_point():
            tensor.div_(len(members))
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    torch.save(dict(model=accumulated, recipe=config['name'], seed=config['seed'], members=members,
                    model_config=identity, step=max(member['step'] for member in members)), temporary)
    temporary.replace(target)
    common.base.write(target.with_suffix('.json'), dict(members=members, weights=[1 / len(members)] * len(members),
                      sha256=common.base.sha(target), parameter_dtype='float32', cross_seed_averaging=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    args = parser.parse_args()
    output = common.HERE / 'models' / args.recipe
    config = json.loads((output / 'run_config.json').read_text())
    assert config['source_sha256'] == common.source_hashes()
    assert json.loads((output / 'TRAIN_COMPLETE.json').read_text())['complete']
    if (output / 'BENCHMARK_COMPLETE.json').exists():
        return
    rows, vocab, contracts, data_root = common.checked_rows()
    assert contracts == config['expected_data']
    members = candidate_members(json.loads((output / 'EPOCHS.json').read_text()))
    paths = {}
    for name, selected in members.items():
        if name == 'best_dev':
            target = Path(selected[0]['path'])
            assert common.base.sha(target) == selected[0]['sha256']
            paths[name] = target
            continue
        target = output / 'averages' / f'{name}.pt'
        if not target.exists():
            average(selected, config, target)
        provenance = json.loads(target.with_suffix('.json').read_text())
        assert provenance['members'] == selected and provenance['sha256'] == common.base.sha(target)
        paths[name] = target
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    model, topo = common.base.build_model(config, vocab, 285)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    model.cuda().eval()
    results = {name: {} for name in paths}
    for split in ('dev', 'test', 'timit_dev', 'timit_test'):
        for name, path in paths.items():
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            model.load_state_dict(saved['model'])
            model.step = saved['step']
            del saved
            gc.collect()
            target = output / 'evaluation' / f'{name}_{split}.json'
            if target.exists():
                summary = json.loads(target.read_text())
                assert summary['checkpoint_sha256'] == common.base.sha(path)
                assert summary['dataset'] == contracts[split]
            else:
                staging = target.with_name(target.stem + '_inprogress.json')
                summary = evaluate(config, model, rows[split], vocab, processor, topo, staging)
                pfer = score_predictions(staging.with_suffix('.predictions.jsonl'), summary)
                summary['pfer'] = pfer
                summary['modes']['text_free']['PFER_percent'] = pfer['PFER_percent']
                summary.update(dataset=contracts[split], checkpoint_sha256=common.base.sha(path),
                               evaluation_uploads=False, candidate=name)
                staging.with_suffix('.predictions.jsonl').replace(target.with_suffix('.predictions.jsonl'))
                common.base.write(target, summary)
            assert summary['utterances'] == len(rows[split])
            assert summary['modes']['forced_alignment']['alignment_failures'] == 0
            results[name][split] = summary
        if split == 'dev':
            selected = min(paths, key=lambda name: results[name]['dev']['modes']['text_free']['nosil_PER_percent'])
            common.base.write(output / 'DEV_SELECTION.json', dict(selected=selected, corpus='Buckeye full40 dev',
                              criterion='lowest FP32 no-sil PER', test_used=False, timit_used=False))
    common.base.write(output / 'FINAL_SUMMARY.json', dict(recipe=args.recipe, results=results, selected=selected))
    common.base.write(output / 'BENCHMARK_COMPLETE.json', dict(recipe=args.recipe, selected=selected,
                      candidates={name: str(path) for name, path in paths.items()}, splits=list(contracts)[1:],
                      evaluation_uploads=False, test_used_for_selection=False, timit_used_for_selection=False))


if __name__ == '__main__':
    main()