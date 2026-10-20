"""Inference-only safetensors averaging with frozen source provenance."""
import importlib.util
import fcntl
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from pfer import HERE, ROOT, sha, write

spec=importlib.util.spec_from_file_location('existing_average',ROOT/'reproduction/average_checkpoints.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.load_model_state=lambda path,mfa=False: load_file(str(path),device='cpu')


def create(objective,steps):
    lock_directory=HERE/'weights'/objective
    lock_directory.mkdir(parents=True,exist_ok=True)
    with (lock_directory/('_'.join(map(str,sorted(steps)))+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _create(objective,steps)


def _create(objective,steps):
    torch.set_num_threads(4)
    steps=sorted(set(steps));assert len(steps)==3
    plan=json.loads((HERE/'plan.json').read_text());pool=plan['pools'][objective]
    candidates={v['step']:v for v in pool['candidates']};assert set(steps)<=set(candidates)
    output=HERE/'weights'/objective/('_'.join(map(str,steps)))/f'checkpoint-{max(steps)}'
    if (output/'AVERAGE_COMPLETE.json').exists():
        provenance=json.loads((output/'AVERAGE_COMPLETE.json').read_text())
        assert provenance['steps']==steps and provenance['objective']==objective
        assert sha(output/'model.safetensors')==provenance['output_sha256']
        with safe_open(str(output/'model.safetensors'),framework='pt',device='cpu') as f:
            metadata=f.metadata() or {}
        if metadata.get('format')!='pt':
            # Older Transformers requires this container metadata to load.
            state=load_file(str(output/'model.safetensors'),device='cpu')
            temp=output/'model.safetensors.tmp';save_file(state,str(temp),metadata={'format':'pt'})
            reread=load_file(str(temp),device='cpu')
            assert set(state)==set(reread) and all(torch.equal(state[k],reread[k]) for k in state)
            temp.replace(output/'model.safetensors')
            provenance.update(output_sha256=sha(output/'model.safetensors'),safetensors_format='pt',
                format_metadata_repacked_with_all_tensors_exactly_equal=True,adapter_sha256=sha(__file__))
            write(output/'AVERAGE_COMPLETE.json',provenance)
        return output
    sources=[]
    for s in steps:
        candidate=candidates[s];d=Path(candidate['checkpoint']);model=d/'model.safetensors'
        assert model.stat().st_size==candidate['model_bytes']
        assert sha(d/'config.json')==candidate['config_sha256']
        sources.append(dict(**candidate,model_sha256=sha(model)))
    output.mkdir(parents=True,exist_ok=True)
    paths=[Path(v['checkpoint'])/'model.safetensors' for v in sources]
    averaged,semantics=helper.average(paths)
    # Independently check sampled entries of every tensor against a literal mean.
    sample={k:torch.linspace(0,v.numel()-1,min(17,v.numel())).long() for k,v in averaged.items() if v.is_floating_point() and v.numel()}
    expected={k:torch.zeros(len(ix),dtype=torch.float64) for k,ix in sample.items()}
    for path in paths:
        state=load_file(str(path),device='cpu')
        for k,ix in sample.items():expected[k]+=state[k].reshape(-1)[ix].double()
        del state
    for k,ix in sample.items():
        torch.testing.assert_close(averaged[k].reshape(-1)[ix],(expected[k]/3).to(averaged[k].dtype),atol=0,rtol=0)
    temp=output/'model.safetensors.tmp';save_file(averaged,str(temp),metadata={'format':'pt'});temp.replace(output/'model.safetensors')
    shutil.copy2(Path(sources[-1]['checkpoint'])/'config.json',output/'config.json')
    provenance=dict(objective=objective,steps=steps,sources=sources,semantics=semantics,
        output_sha256=sha(output/'model.safetensors'),config_sha256=sha(output/'config.json'),
        average_helper_sha256=sha(ROOT/'reproduction/average_checkpoints.py'),adapter_sha256=sha(__file__),
        sampled_tensor_arithmetic_verified=True,safetensors_format='pt',optimizer_updated=False)
    write(output/'AVERAGE_COMPLETE.json',provenance)
    print('AVERAGED',objective,steps,str(output),flush=True)
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--objective',choices=['ce','jsd_upstream','jsd_standard'],required=True)
    args=parser.parse_args();plan=json.loads((HERE/'plan.json').read_text())
    create(args.objective,plan['pools'][args.objective]['last3'])
