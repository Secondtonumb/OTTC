"""Compare real continuous/resumed GPU model and optimizer state."""
import json
from pathlib import Path
import sys
import torch
import numpy as np


def difference(a,b):
    if isinstance(a,np.ndarray):
        assert np.array_equal(a,b)
        return 0.
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
        return float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max((difference(x,y) for x,y in zip(a,b)),default=0.)
    assert a==b,(a,b)
    return 0.


root=Path(sys.argv[1]);result={}
for cost in ('jsd','ce'):
    a=torch.load(root/f'{cost}_continuous/resume.pt',map_location='cpu',weights_only=False)
    b=torch.load(root/f'{cost}_resume/resume.pt',map_location='cpu',weights_only=False)
    result[cost]={k:difference(a[k],b[k]) for k in ('model','model_avg','optimizer','scheduler','rank_states')}
    assert all(v==0 for v in result[cost].values()),result
    assert a['progress']['global_step']==b['progress']['global_step']==4
(root/'RESUME_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
