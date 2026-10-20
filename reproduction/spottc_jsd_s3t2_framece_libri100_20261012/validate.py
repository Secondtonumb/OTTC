"""Compare masks, rounding, raw CE and gradients with the historical implementation."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from framece_model import targets
from prepare import sha, HERE


def main():
    old = Path('/home/m64000/work/IF-MDD/models/phn_mono_ssl_model_v3_refactored.py')
    fn = next(n for n in ast.walk(ast.parse(old.read_text())) if isinstance(n,ast.FunctionDef)
              and n.name=='_compute_training_alignment_ce_loss')
    fn.decorator_list=[]
    namespace={'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(old),'exec'),namespace)
    rows=[dict(duration=1.03,intervals=[[3,.015,.21],[14,.21,.555],[103,.70,1.03]]),
          dict(duration=.82,intervals=[[27,.04,.399],[41,.399,.67]])]
    lengths=torch.tensor([103,82]);gold=targets(rows,lengths,torch.device('cpu'))
    probs=torch.randn(2,103,105,requires_grad=True).log_softmax(-1)
    labels=torch.tensor([[3,14,103],[27,41,-100]])
    starts=torch.tensor([[.015,.21,.70],[.04,.399,0.]])
    ends=torch.tensor([[.21,.555,1.03],[.399,.67,0.]])
    shim=SimpleNamespace(use_training_alignment_ce=True,training_alignment_ce_weight=1.,
        training_alignment_ce_ignore_index=-100,_unpack_padded_tensor=lambda x:(x,None),
        _alignment_ce_duration_seconds=lambda *args:torch.tensor([1.03,.82]))
    batch=SimpleNamespace(alignment_ce_labels=labels,alignment_ce_starts=starts,
                          alignment_ce_ends=ends,alignment_ce_duration=torch.tensor([1.03,.82]))
    historical=namespace[fn.name](shim,probs,batch,lengths/103.)
    current=torch.nn.functional.nll_loss(probs.reshape(-1,105),gold.reshape(-1),ignore_index=-100)
    torch.testing.assert_close(current,historical,rtol=0,atol=0)
    grad1=torch.autograd.grad(current,probs,retain_graph=True)[0]
    grad2=torch.autograd.grad(historical,probs)[0]
    torch.testing.assert_close(grad1,grad2,rtol=0,atol=0)
    assert not bool(grad1[gold==-100].any())
    (HERE/'validation/CE_REPLAY_PASS.json').write_text(json.dumps(dict(
        historical_loss_source=str(old),historical_loss_source_sha256=sha(old),
        exact_loss_replay=True,exact_gradient_replay=True,valid_frames=int((gold!=-100).sum()),
        loss=float(current.detach())))+'\n')
    print('CE_REPLAY_PASS',flush=True)


if __name__=='__main__':main()
