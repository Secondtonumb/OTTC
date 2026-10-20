"""Run the frozen smoke with detailed state oracles in a separate directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

HERE=Path(__file__).resolve().parent
RECIPE=HERE.parent
sys.path.insert(0,str(RECIPE))
import runtime
import torch


def compare(a,b,path='state'):
    differences=[]
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape!=b.shape or a.dtype!=b.dtype:
            return [dict(path=path,kind='tensor_metadata')]
        if not torch.equal(a,b):
            af,bf=a.double(),b.double()
            differences.append(dict(path=path,kind='tensor',dtype=str(a.dtype),
                max_abs=float((af-bf).abs().max()),elements=a.numel(),
                unequal=int((a!=b).sum()),finite=bool(torch.isfinite(af).all() and torch.isfinite(bf).all())))
    elif isinstance(a,dict):
        if set(a)!=set(b):return [dict(path=path,kind='dictionary_keys')]
        for k in a:differences.extend(compare(a[k],b[k],path+'/'+str(k)))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):return [dict(path=path,kind='sequence_length')]
        for i,(x,y) in enumerate(zip(a,b)):differences.extend(compare(x,y,path+'/'+str(i)))
    else:
        import numpy as np
        equal=np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b
        if not equal:differences.append(dict(path=path,kind='value',expected=repr(a)[:160],actual=repr(b)[:160]))
    return differences


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--deterministic',action='store_true')
    args=parser.parse_args()
    scope=HERE/('deterministic' if args.deterministic else 'original')
    scope.mkdir(parents=True,exist_ok=True)
    if args.deterministic:torch.use_deterministic_algorithms(True)
    module=runpy.run_path(str(RECIPE/'validate.py'),run_name='frozen_smoke_diagnostic')
    globals_=module['main'].__globals__
    globals_['HERE']=scope
    original_digest=globals_['digest']
    import checkpoint
    index=0
    def record_digest(value):
        nonlocal index
        if isinstance(value,dict) and {'model','optimizer','random','scaler','sampler'}.issubset(value):
            torch.save(checkpoint.tree_to(value,'cpu'),scope/(('resume' if args.resume else 'initial')+f'-state-{index}.pt'))
            index+=1
        return original_digest(value)
    globals_['digest']=record_digest
    sys.argv=[str(RECIPE/'validate.py')]+(['--resume-phase'] if args.resume else [])
    error=None
    try:module['main']()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    report=dict(resume=args.resume,deterministic=args.deterministic,error=error)
    if args.resume:
        for i,label in [(0,'restored'),(1,'continued')]:
            expected,actual=scope/f'initial-state-{i}.pt',scope/f'resume-state-{i}.pt'
            if expected.exists() and actual.exists():
                diff=compare(torch.load(expected,weights_only=False),torch.load(actual,weights_only=False))
                report[label]=dict(differing_entries=len(diff),differences=diff)
    (scope/('resume_report.json' if args.resume else 'initial_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
