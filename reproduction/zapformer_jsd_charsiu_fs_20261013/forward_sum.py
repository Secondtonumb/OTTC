"""Configurable Charsiu criterion using the existing phone-state output."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from charsiu_forward_sum import ForwardSumLoss
from ifmdd_losses import topology_forward_sum_loss


def validate_config(config):
    defaults = dict(ottc_weight=1., forward_sum_weight=1., forward_sum_kind='charsiu_positions',
                    forward_sum_normalization='target', forward_sum_reduction='mean',
                    forward_sum_blank_logprob=-1., forward_sum_temperature=1.,
                    forward_sum_zero_infinity=True, forward_sum_ctc_device='cpu', forward_sum_output_beam=10.)
    if set(config)-set(defaults):raise ValueError('Unknown loss config: '+str(set(config)-set(defaults)))
    result=defaults|config
    for key in ('ottc_weight','forward_sum_weight','forward_sum_blank_logprob','forward_sum_temperature','forward_sum_output_beam'):
        if not math.isfinite(result[key]):raise ValueError(key+' must be finite')
    if min(result['ottc_weight'],result['forward_sum_weight'])<0 or result['ottc_weight']+result['forward_sum_weight']<=0:
        raise ValueError('Loss weights must be nonnegative and at least one positive')
    if result['forward_sum_temperature']<=0 or result['forward_sum_output_beam']<=0:
        raise ValueError('Temperature and output beam must be positive')
    if result['forward_sum_kind'] not in ('charsiu_positions','native_topology'):raise ValueError('Unknown forward-sum kind')
    if result['forward_sum_normalization'] not in ('target','input','none'):raise ValueError('Unknown normalization')
    if result['forward_sum_reduction'] not in ('mean','sum'):raise ValueError('Unknown reduction')
    if result['forward_sum_ctc_device'] not in ('cpu','cuda'):raise ValueError('Unknown CTC device')
    if not isinstance(result['forward_sum_zero_infinity'],bool):raise ValueError('zero_infinity must be boolean')
    if result['forward_sum_kind']=='native_topology' and (result['forward_sum_normalization'],result['forward_sum_reduction'])!=('input','sum'):
        raise ValueError('Native topology comparison uses its unchanged input-normalized sum')
    return result


def position_log_probs(logits,tokens,token_lengths,temperature=1.):
    """Merge S0/S1/S2 probabilities; gather every transcript occurrence separately."""
    if logits.shape[-1]!=313:raise ValueError('Expected unchanged 313-state output')
    if bool((token_lengths<=0).any()) or int(token_lengths.sum())!=tokens.numel():raise ValueError('Invalid target lengths')
    if bool(((tokens<3)|(tokens>104)).any()):raise ValueError('Targets must be valid non-control phone IDs 3..104')
    logp=logits.float().log_softmax(-1)
    phones=torch.cat([logp[...,:1],torch.logsumexp(logp[...,1:].reshape(*logp.shape[:-1],104,3),-1)],-1)
    sequences=torch.split(tokens,token_lengths.tolist())
    padded=nn.utils.rnn.pad_sequence(sequences,batch_first=True,padding_value=0)
    selected=phones.gather(-1,padded[:,None,:].expand(-1,logits.shape[1],-1))/temperature
    valid=torch.arange(padded.shape[1],device=logits.device)[None,:]<token_lengths[:,None]
    return selected.masked_fill(~valid[:,None,:],-torch.inf).log_softmax(-1).unsqueeze(1)


class ConfigurableForwardSum(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=validate_config(config)
        self.charsiu=ForwardSumLoss(blank_logprob=self.config['forward_sum_blank_logprob'])
        self.charsiu.CTCLoss=nn.CTCLoss(zero_infinity=self.config['forward_sum_zero_infinity'])

    def forward(self,logits,lengths,tokens,token_lengths,topology):
        c=self.config
        if c['forward_sum_kind']=='native_topology':
            return topology_forward_sum_loss(logits.log_softmax(-1),lengths,tokens,token_lengths,
                    topology=topology,blank_id=0,output_beam=c['forward_sum_output_beam'])
        scores=position_log_probs(logits,tokens,token_lengths,c['forward_sum_temperature'])
        # CPU CTC keeps the original algorithm and differentiable device copy,
        # while avoiding PyTorch CUDA CTC's nondeterministic backward kernel.
        device=torch.device(c['forward_sum_ctc_device'])
        scores=scores.to(device);text_lengths=token_lengths.to(device);frame_lengths=lengths.to(device)
        if c['forward_sum_normalization']=='target' and c['forward_sum_reduction']=='mean':
            return self.charsiu(scores,text_lengths,frame_lengths).to(logits.device)
        padded=F.pad(scores,(1,0),value=c['forward_sum_blank_logprob'])
        costs=[]
        for i in range(scores.shape[0]):
            n=int(text_lengths[i]);t=int(frame_lengths[i])
            logp=padded[i,0,:t,:n+1].log_softmax(-1).unsqueeze(1)
            labels=torch.arange(1,n+1,device=device).unsqueeze(0)
            value=F.ctc_loss(logp,labels,frame_lengths[i:i+1],text_lengths[i:i+1],
                             reduction='none',zero_infinity=c['forward_sum_zero_infinity'])[0]
            if c['forward_sum_normalization']=='target':value=value/n
            elif c['forward_sum_normalization']=='input':value=value/t
            costs.append(value)
        values=torch.stack(costs)
        return (values.mean() if c['forward_sum_reduction']=='mean' else values.sum()).to(logits.device)
