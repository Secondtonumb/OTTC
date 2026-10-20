"""Monophone inventory adapters; frozen native Zapformer and upstream OTTC are reused."""
from pathlib import Path
import sys

NATIVE=Path(__file__).resolve().parent.parent/'zapformer_s3t2ss_native_20261012'
sys.path.insert(0,str(NATIVE))
import runtime
import torch
from torch import nn
from task_model import S3T2,parameters
from zapformer_modules import ScaledLinear
import train as native
from ifmdd_losses import pad_flat_targets,make_ottc_inputs
from archived_ot_loss import batched_ottc_loss_bucketized


def merge_states(logits):
    if logits.shape[-1]!=313:raise ValueError('Pooling requires the unchanged 313-state head')
    # logsumexp on raw logits is exactly equivalent to summing normalized
    # substate probabilities and renormalizing the 105 phone probabilities.
    logits=logits.float()
    return torch.cat([logits[...,:1],torch.logsumexp(logits[...,1:].reshape(*logits.shape[:-1],104,3),-1)],-1)


class Monophone(S3T2):
    def __init__(self,variant='phone105',tiny=False):
        if variant not in ('phone105','pooled313'):raise ValueError(variant)
        super().__init__('jsd',tiny=tiny,topology='s3_t2_star_star')
        self.variant=variant
        self.topology='s1_t1'
        if variant=='phone105':
            # Keep the canonical native seed42 backbone and mass initialization;
            # replace only the output head with a genuinely 105-dimensional one.
            dim=native.lookup(parameters(tiny),'embed_dim')
            self.acoustic.ctc_output=nn.Sequential(ScaledLinear(dim,105,initial_scale=.1))
        self.loss_config=dict(ottc_cost='jsd_upstream',ottc_weight=1.,forward_sum_weight=0.,
                              target_states_per_phone=1,head_dimension=105 if variant=='phone105' else 313)

    def phone_logits(self,encoder_output):
        logits=self.acoustic.ctc_output(encoder_output).float()
        if self.variant=='pooled313':logits=merge_states(logits)
        return logits.index_fill(-1,torch.tensor([1,2],device=logits.device),-1e4)

    def forward(self,features,lengths,tokens,token_lengths,frame_targets=None,aux_loss_scale=0.):
        if bool(((tokens<3)|(tokens>104)).any()):raise ValueError('Only native phone IDs 3..104 are valid targets')
        enc,out_lengths=self.acoustic.forward_encoder(features,lengths,aux_loss_scale=aux_loss_scale)
        assert enc.shape[1]==features.shape[1] and torch.equal(lengths,out_lengths)
        logits=self.phone_logits(enc)
        with torch.autocast(features.device.type,enabled=False):
            padded=pad_flat_targets(tokens,token_lengths)
            weights=self.weight_head(enc.float()).squeeze(-1)
            fw,tw,fm,onehot=make_ottc_inputs(logits,out_lengths,padded,token_lengths,weights)
            tm=torch.arange(padded.shape[1],device=logits.device)[None]<token_lengths[:,None]
            ids=torch.cat([torch.zeros(1,device=logits.device,dtype=torch.long),torch.arange(3,105,device=logits.device)])
            # Exact archived upstream implementation; only inventory/target
            # representation change. No replacement divergence or mass rule.
            loss=batched_ottc_loss_bucketized(logits.index_select(-1,ids),onehot.index_select(-1,ids),
                                             fw,tw,amask=fm,bmask=tm,jsd=True)[0]
            return loss,dict(ottc_sum=loss.detach(),jsd_sum=loss.detach(),forward_sum_raw=loss.detach().new_zeros(()))
