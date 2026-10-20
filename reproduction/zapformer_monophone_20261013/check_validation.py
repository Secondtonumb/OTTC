"""Full eight-rank bitwise checkpoint equivalence for the new true105 adapter."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'zapformer_s3t2ss_native_20261012'))
import runtime
import torch
import numpy as np


def identical(a,b,path):
    if torch.is_tensor(a):
        assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a:identical(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):identical(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


def main():
    root=HERE/'validation_ddp'
    a=torch.load(root/'uninterrupted/step-000000002.pt',map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(root/'resumed/step-000000002.pt',map_location='cpu',mmap=True,weights_only=False)
    for key in ['model','model_avg','optimizer','scheduler','rank_states','contract']:
        identical(a[key],b[key],key)
    assert a['world_size']==b['world_size']==8
    assert a['progress']['optimizer_updates']==b['progress']['optimizer_updates']==2
    assert a['model']['acoustic.ctc_output.0.weight'].shape[0]==105
    for name in ['uninterrupted','resumed']:
        rows=[json.loads(line) for line in (root/name/'updates.jsonl').read_text().splitlines()]
        assert [r['optimizer_step'] for r in rows]==[1,2]
        assert all(r['utterances']>1 and r['optimizer_updates']==r['optimizer_step'] for r in rows)
        assert all(r['forward_sum_raw']==0 for r in rows)
    result=dict(world_size=8,full_native_model=True,head_dimension=105,optimizer_steps=2,
                model_optimizer_average_scheduler_all_rank_rng_bitwise_equal=True,
                original_ottc_jsd=True,forward_sum_weight=0.,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [HERE/'train_native.py',HERE/'monophone_model.py',HERE/'monophone_metrics.py']})
    (HERE/'DDP_RESUME_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
