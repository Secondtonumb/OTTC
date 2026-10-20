"""Replay which acoustic dimensions the approved Charsiu score adaptation supervises."""
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'zapformer_s3t2ss_native_20261012'))
import runtime
import torch
sys.path.insert(0,str(HERE.parent/'zapformer_libri100_ablation_20261013'))
from forward_sum import ConfigurableForwardSum

torch.manual_seed(42)
config=json.loads((HERE.parent/'zapformer_libri100_ablation_20261013/loss_config.json').read_text())
x=torch.randn(2,20,313,requires_grad=True)
tokens=torch.tensor([3,4,3,7,8]);token_lengths=torch.tensor([3,2]);lengths=torch.tensor([20,18])
criterion=ConfigurableForwardSum(config)
loss=criterion(x,lengths,tokens,token_lengths,'s3_t2_star_star')
gradient=torch.autograd.grad(loss,x)[0]
blank=x.detach().clone();blank[:,:,0]+=20
phase=x.detach().clone()
phase[:,:,1:]=phase[:,:,1:].reshape(2,20,104,3).flip(-1).reshape(2,20,312)
loss_blank=criterion(blank,lengths,tokens,token_lengths,'s3_t2_star_star')
loss_phase=criterion(phase,lengths,tokens,token_lengths,'s3_t2_star_star')
torch.testing.assert_close(loss,loss_blank,rtol=1e-6,atol=1e-6)
torch.testing.assert_close(loss,loss_phase,rtol=1e-6,atol=1e-6)
assert gradient[:,:,0].abs().max()<1e-7
result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_loss=float(loss),loss_after_blank_logit_plus20=float(loss_blank),
            loss_after_substate_reversal=float(loss_phase),max_absolute_blank_gradient=float(gradient[:,:,0].abs().max()),
            blank_probability_inside_position_CTC=float(torch.sigmoid(torch.tensor(-1.))),
            conclusion='Merged-position auxiliary is insensitive to acoustic blank and substate ordering; OTTC still supervises states')
(HERE/'forward_sum_invariance.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
