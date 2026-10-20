"""Verify sparse OT numerics, four-head gradients and padding semantics."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT)]
import torch
from torch.nn import functional as F
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from transformers import Wav2Vec2Config
from multihead import HEADS,MultiPhoneConfig,MultiPhoneModel,transport_loss
from ottc.ops_loss.ot_loss import batched_ottc_loss_bucketized


def main():
    torch.manual_seed(42);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.randn(2,8,5,device=device,requires_grad=True)
    a=torch.randn(2,8,device=device,requires_grad=True)
    length=torch.tensor([8,6],device=device)
    mask=torch.arange(8,device=device)[None]<length[:,None]
    weights=a.masked_fill(~mask,-torch.inf).softmax(-1)
    target=torch.tensor([[1,2,3],[2,1,-100]],device=device)
    for cost,js in [('ce',False),('jsd_upstream',True),('jsd_standard','standard')]:
        result=transport_loss(x,target,weights,length,cost)
        assert torch.isfinite(result)
        if device=='cuda':
            label_mask=(target>=0).float()
            onehot=F.one_hot(target.clamp_min(0),5)
            expected=batched_ottc_loss_bucketized(x,onehot,weights,label_mask/label_mask.sum(-1)[:,None],
                                                amask=mask,bmask=label_mask,jsd=js)[0]/2
            torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-6)
        result.backward(retain_graph=True)
        assert torch.isfinite(x.grad).all() and torch.isfinite(a.grad).all()
        x.grad=None;a.grad=None
    base=Wav2Vec2Config(hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,
                       conv_dim=(4,4,4),conv_stride=(5,4,4),conv_kernel=(10,8,4),
                       num_conv_pos_embeddings=8,num_conv_pos_embedding_groups=2,mask_time_prob=0.)
    vocabs={h:{'<blank>':0,'a':1,'b':2,'c':3,'d':4} for h in HEADS}
    for objective in ['ctc','ce','jsd_upstream']:
        model=MultiPhoneModel(MultiPhoneConfig(encoder_config=base.to_dict(),vocabs=vocabs,objective=objective)).to(device)
        audio=torch.randn(2,1600,device=device)
        amask=torch.ones_like(audio,dtype=torch.long);amask[1,1400:]=0
        labels={h:target for h in HEADS}
        result=model(audio,amask,labels)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        for h in HEADS:
            assert model.phone_heads[h].weight.grad is not None
            assert torch.isfinite(model.phone_heads[h].weight.grad).all()
            if objective!='ctc':assert model.mass_heads[h].weight.grad is not None
        assert model.encoder.encoder.layers[0].attention.q_proj.weight.grad is not None
        assert all(p.grad is None for p in model.encoder.feature_extractor.parameters())
        partial={h:target.clone() for h in HEADS}
        partial['mfa_ipa'][1]=-100
        model.zero_grad();masked=model(audio,amask,partial)
        assert torch.isfinite(masked['loss'])
        masked['loss'].backward()
        partial['mfa_ipa'][:]=-100
        model.zero_grad();masked=model(audio,amask,partial)
        assert torch.isfinite(masked['loss']) and masked['head_losses']['mfa_ipa']==0
        masked['loss'].backward()
        if objective!='ctc':
            model.freeze_alignment()
            assert all(not p.requires_grad for p in model.mass_heads.parameters())
            assert all(not p.requires_grad for p in model.alignment_encoder.parameters())
            model.zero_grad();model(audio,amask,labels)['loss'].backward()
            assert model.phone_heads['mfa_ipa'].weight.grad is not None
        print('PASS',objective,'four classification-head gradients, shared encoder gradients, padding, frozen CNN/alignment',flush=True)
    print('PASS sparse OT matches upstream on GPU' if device=='cuda' else 'CPU PASS; GPU upstream equivalence still pending',flush=True)


if __name__=='__main__':main()
