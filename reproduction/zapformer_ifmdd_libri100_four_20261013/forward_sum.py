"""Use original IF-MDD transcript graph and k2 raw losses, with explicit reduction."""
import math
import torch
from utils.subphonetic_topology import SubphoneticTopologyLoss
from task_ottc import HERE, relative_lengths
from ifmdd_losses import pad_flat_targets


def validate_config(config):
    required={'arm','ottc_cost','ottc_weight','forward_sum_weight','forward_sum_kind',
              'forward_sum_normalization','forward_sum_reduction','forward_sum_output_beam','topology'}
    if set(config)!=required:raise ValueError('Incomplete/unknown loss config')
    if config['ottc_cost'] not in ('ce','ifmdd_standard_jsd'):raise ValueError('Unknown IF-MDD cost')
    if config['forward_sum_kind']!='ifmdd_s3t2ss' or config['topology']!='s3_t2_star_star':raise ValueError('S3T2** required')
    if (config['forward_sum_normalization'],config['forward_sum_reduction'])!=('input','sum'):raise ValueError('Requires input-normalized sum')
    for key in ['ottc_weight','forward_sum_weight','forward_sum_output_beam']:
        if not math.isfinite(config[key]) or config[key]<0:raise ValueError(key)
    if config['ottc_weight']<=0 or config['forward_sum_output_beam']<=0:raise ValueError('Positive OTTC weight/output beam required')
    return dict(config)


class IFMDDForwardSum(SubphoneticTopologyLoss):
    def __init__(self,config):
        super().__init__(str(HERE/'mfa_label_encoder.txt'),context_phone_mode='mono',
                         topology='s3_t2_star_star',split_silence=False,blank_index=0,
                         loss_type='ctc',backend='k2',reduction='none',
                         k2_output_beam=config['forward_sum_output_beam'],
                         repeat_separator='optional_blank',silence_self_loops='none')

    def forward(self,logits,lengths,tokens,token_lengths):
        targets=pad_flat_targets(tokens,token_lengths)
        per_loss=super().forward(logits.float().log_softmax(-1),targets,
                                 relative_lengths(lengths,logits.shape[1]),
                                 relative_lengths(token_lengths,targets.shape[1]))
        effective=(relative_lengths(lengths,logits.shape[1])*logits.shape[1]).long().clamp(1,logits.shape[1])
        if not torch.isfinite(per_loss).all():raise ValueError('No finite S3T2** forward-sum path')
        return (per_loss/effective).sum().float(),per_loss.sum().float()
