"""Same frozen TIMIT scorer/PFER; headline is single best-dev, averages separate."""
import gc
import json

import torch

from reproduction.crottc_ag_seed_sweep_20261022.common import sha,write
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def average(paths,output,config):
    if output.exists():
        return
    accumulated=None
    members=[]
    for path in paths:
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        assert saved['recipe']==config['name']
        members.append(dict(path=str(path),epoch=saved['epoch'],step=saved['step'],sha256=sha(path)))
        if accumulated is None:
            accumulated={k:v.float().clone() if v.is_floating_point() else v.clone() for k,v in saved['model'].items()}
        else:
            for key,value in saved['model'].items():
                if value.is_floating_point():
                    assert value.dtype==torch.float32
                    accumulated[key].add_(value)
                else:
                    assert torch.equal(accumulated[key],value)
    for value in accumulated.values():
        if value.is_floating_point():
            value.div_(len(paths))
            assert value.dtype==torch.float32
    temporary=output.with_suffix('.tmp')
    torch.save(dict(model=accumulated,recipe=config['name'],members=members,model_config=saved['model_config']),temporary)
    temporary.replace(output)
    write(output.with_suffix('.json'),dict(members=members,sha256=sha(output),dtype='float32'))


def final_benchmarks(config,model,rows,vocab,processor,topo,output,run):
    if (output/'BENCHMARK_COMPLETE.json').exists():
        return
    completed=json.loads((output/'TRAIN_COMPLETE.json').read_text())
    epoch=int(round(completed['epoch']))
    assert epoch==40
    candidates=dict(best_dev=output/'best_dev.pt',final=output/'epoch_weights'/f'epoch-{epoch:03d}.pt')
    for count in (5,10):
        path=output/f'final_{count}_average.pt'
        average([output/'epoch_weights'/f'epoch-{e:03d}.pt' for e in range(epoch-count+1,epoch+1)],path,config)
        candidates[f'final_{count}']=path
    results={}
    model.cuda()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    for split in ('dev','test'):
        for name,path in candidates.items():
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            model.load_state_dict(saved['model'])
            model.step=saved.get('step',completed['step'])
            del saved
            gc.collect()
            target=output/f'benchmark_{name}_{split}.json'
            summary=json.loads(target.read_text()) if target.exists() else evaluate(config,model,rows[split],vocab,processor,topo,target)
            summary['pfer']=score_predictions(target.with_suffix('.predictions.jsonl'),summary)
            summary['modes']['text_free']['PFER_percent']=summary['pfer']['PFER_percent']
            write(target,summary)
            assert summary['modes']['forced_alignment']['alignment_failures']==0
            results.setdefault(name,{})[split]=summary
        if split=='dev':
            selected=min(candidates,key=lambda n:results[n]['dev']['modes']['text_free']['nosil_PER_percent'])
            write(output/'DEV_SELECTION.json',dict(selected=selected,test_used=False,
                criterion='lowest dev PER among predetermined best/final/final5/final10; four-seed headline always uses single best_dev',
                comparison_candidate='best_dev'))
    write(output/'BENCHMARK_COMPLETE.json',dict(selected=selected,test_used_for_selection=False,comparison_candidate='best_dev',
        candidates={name:dict(path=str(path),sha256=sha(path)) for name,path in candidates.items()}))
    write(output/'FINAL_SUMMARY.json',dict(seed=config['seed'],cost=config['cost'],comparison_candidate='best_dev',
        dev=results['best_dev']['dev'],test=results['best_dev']['test']))
