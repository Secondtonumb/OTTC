"""MS-TCN temporal log-probability smoothing for variable-length speech.

Reference: https://github.com/yabufarha/ms-tcn/blob/master/model.py
Retains the detached previous frame and the truncated per-class square loss.
Unlike the video implementation, padding never contributes to normalization.
"""
import math

import torch
import torch.nn.functional as F


def temporal_smoothing_loss(logits, valid_mask, tau=4.0, reduction='mean'):
    """Smooth all classes of ``[B,T,C]`` logits without boundary targets.

    ``mean`` is the mean over every valid adjacent frame pair and class, as
    in the requested speech adaptation. ``none`` returns one valid-pair/class
    mean per utterance. ``sum`` sums these utterance means, matching OTTC's
    batch-summed objective and allowing exact microbatch accumulation.
    Utterances with fewer than two valid adjacent frames contribute zero.
    """
    if logits.ndim != 3 or logits.shape[-1] == 0:
        raise ValueError('logits must have shape [B,T,C] with C > 0')
    if not logits.is_floating_point():
        raise ValueError('logits must be floating point')
    if valid_mask.shape != logits.shape[:2]:
        raise ValueError('valid_mask must match the output-frame shape [B,T]')
    if valid_mask.device != logits.device:
        raise ValueError('valid_mask and logits must be on the same device')
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(tau * tau):
        raise ValueError('tau must be finite and positive with finite tau squared')
    if reduction not in ('mean', 'none', 'sum'):
        raise ValueError('reduction must be mean, none or sum')

    valid = valid_mask.bool()
    # Sanitize padding before log_softmax: even NaN padding must have zero
    # influence/gradient. No valid frame is changed by this operation.
    safe_logits = logits.float().masked_fill(~valid.unsqueeze(-1), 0.)
    logp = F.log_softmax(safe_logits, dim=-1)
    error = (logp[:, 1:] - logp[:, :-1].detach()).square().clamp(max=tau * tau)
    valid_pair = valid[:, 1:] & valid[:, :-1]
    error = error.masked_fill(~valid_pair.unsqueeze(-1), 0.)
    numerator = error.sum(dim=(1, 2))
    pair_count = valid_pair.sum(dim=1)
    if reduction == 'mean':
        return numerator.sum() / (pair_count.sum().clamp_min(1) * logits.shape[-1])
    per_utterance = numerator / (pair_count.clamp_min(1) * logits.shape[-1])
    return per_utterance if reduction == 'none' else per_utterance.sum()
