"""Batch-local OTTC vocabulary with full normalization and chunked backward.

Design reference: yfyeung/PrunedCTC, e60479249598274a0ccb7b93074d40ab95090a11.
No CTC blank state, beam pruning, or subset renormalization is introduced.
"""

from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as functional


@dataclass(frozen=True)
class BatchVocabulary:
    selected: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor


def remap_targets(targets, vocab_size, *, blank_id=0):
    if targets.ndim != 2 or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError('targets must be a padded int32/int64 matrix')
    valid = targets != -100
    if targets.shape[0] == 0 or not bool(valid.any(dim=1).all()):
        raise ValueError('each utterance needs at least one target')
    if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError('targets must be right-padded with -100')
    values = targets[valid].long()
    if bool(((values < 0) | (values >= vocab_size)).any()):
        raise ValueError('target outside the full vocabulary')
    if blank_id is not None and bool((values == blank_id).any()):
        raise ValueError('OTTC targets must not contain blank')
    selected, inverse = torch.unique(values, sorted=True, return_inverse=True)
    local = torch.full_like(targets, -100, dtype=torch.int64)
    local[valid] = inverse
    return BatchVocabulary(selected, local, valid.sum(-1))


def _projection(hidden, weight, bias, dtype):
    return functional.linear(hidden, weight.to(dtype), None if bias is None else bias.to(dtype))


def _background_terms(probabilities, jsd_function):
    with torch.enable_grad():
        independent = probabilities.detach().requires_grad_(True)
        values = jsd_function(independent, torch.zeros_like(independent, dtype=torch.float32))
        slopes, = torch.autograd.grad(values.sum(), independent)
    return values.detach(), slopes.detach()


class _ReducedLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, selected, chunk_size, jsd_function):
        dtype = torch.float64 if torch.float64 in (hidden.dtype, weight.dtype) else torch.float32
        flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype)
        normalizer = torch.full((len(flat),), -torch.inf, device=hidden.device, dtype=torch.float64)
        for start in range(0, len(weight), chunk_size):
            stop = min(start + chunk_size, len(weight))
            logits = _projection(flat, weight[start:stop], None if bias is None else bias[start:stop], dtype)
            normalizer = torch.logaddexp(normalizer, logits.double().logsumexp(-1))
        del logits
        selected_logits = _projection(flat, weight.index_select(0, selected),
            None if bias is None else bias.index_select(0, selected), dtype)
        log_probs = (selected_logits.double() - normalizer[:, None]).to(dtype)
        background = torch.zeros_like(normalizer)
        mean_slope = torch.zeros_like(normalizer)
        if jsd_function is not None:
            for start in range(0, len(weight), chunk_size):
                stop = min(start + chunk_size, len(weight))
                logits = _projection(flat, weight[start:stop],
                    None if bias is None else bias[start:stop], dtype)
                probabilities = (logits.double() - normalizer[:, None]).exp().to(dtype)
                values, slopes = _background_terms(probabilities, jsd_function)
                background.add_(values.double().sum(-1))
                mean_slope.add_((probabilities.double() * slopes.double()).sum(-1))
        ctx.save_for_backward(hidden, weight, bias, selected, normalizer, log_probs, mean_slope)
        ctx.chunk_size = chunk_size
        ctx.dtype = dtype
        ctx.jsd_function = jsd_function
        ctx.set_materialize_grads(False)
        return (log_probs.reshape(*hidden.shape[:-1], len(selected)),
                background.to(dtype).reshape(hidden.shape[:-1]))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_background):
        hidden, weight, bias, selected, normalizer, saved_log_probs, mean_slope = ctx.saved_tensors
        with torch.autocast(hidden.device.type, enabled=False):
            flat = hidden.reshape(-1, hidden.shape[-1]).to(ctx.dtype)
            gradient = (torch.zeros_like(saved_log_probs) if grad_output is None else
                        grad_output.reshape(-1, len(selected)).to(ctx.dtype))
            negative_sum = -gradient.sum(-1, keepdim=True)
            need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
            grad_hidden = torch.zeros_like(flat) if need_hidden else None
            grad_weight = torch.empty_like(weight) if need_weight else None
            grad_bias = torch.empty_like(bias) if need_bias and bias is not None else None
            for start in range(0, len(weight), ctx.chunk_size):
                stop = min(start + ctx.chunk_size, len(weight))
                logits = _projection(flat, weight[start:stop],
                    None if bias is None else bias[start:stop], ctx.dtype)
                probabilities = (logits.double() - normalizer[:, None]).exp().to(ctx.dtype)
                background_gradient = None
                if ctx.jsd_function is not None and grad_background is not None:
                    _, slopes = _background_terms(probabilities, ctx.jsd_function)
                    background_gradient = (probabilities.double() * (slopes.double() - mean_slope[:, None])
                        * grad_background.reshape(-1, 1).double()).to(ctx.dtype)
                grad_logits = probabilities
                positions = ((selected >= start) & (selected < stop)).nonzero().flatten()
                local = selected.index_select(0, positions) - start
                grad_logits.index_copy_(1, local, saved_log_probs.index_select(1, positions).exp())
                grad_logits.mul_(negative_sum)
                grad_logits.index_add_(1, local, gradient.index_select(1, positions))
                if background_gradient is not None:
                    grad_logits.add_(background_gradient)
                if need_hidden:
                    grad_hidden.addmm_(grad_logits, weight[start:stop].to(ctx.dtype))
                if need_weight:
                    grad_weight[start:stop] = grad_logits.t().mm(flat).to(weight.dtype)
                if grad_bias is not None:
                    grad_bias[start:stop] = grad_logits.sum(0).to(bias.dtype)
            return (None if grad_hidden is None else grad_hidden.reshape_as(hidden).to(hidden.dtype),
                    grad_weight, grad_bias, None, None, None)


def _reduced_statistics(hidden, weight, bias, selected, chunk_size, jsd_function):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError('hidden and projection dimensions do not match')
    if hidden.numel() == 0 or weight.numel() == 0:
        raise ValueError('hidden and weight must be nonempty')
    if bias is not None and bias.shape != (len(weight),):
        raise ValueError('bias must have one entry per vocabulary class')
    for value in (hidden, weight, bias):
        if value is not None:
            if value.device != hidden.device or value.dtype not in (torch.float32, torch.float64, torch.bfloat16):
                raise ValueError('projection inputs must share a device and use float32, float64, or bfloat16')
    if selected.ndim != 1 or selected.numel() == 0 or selected.dtype != torch.int64:
        raise ValueError('selected must be a nonempty int64 vector')
    if selected.device != hidden.device or bool(((selected < 0) | (selected >= len(weight))).any()):
        raise ValueError('selected IDs must be on the hidden device and inside the vocabulary')
    if selected.unique().numel() != selected.numel():
        raise ValueError('selected IDs must be unique')
    if torch.is_autocast_enabled(hidden.device.type):
        raise RuntimeError('disable autocast around reduced_log_softmax')
    return _ReducedLogSoftmax.apply(hidden, weight, bias, selected, chunk_size, jsd_function)


def reduced_log_softmax(hidden, weight, bias, selected, *, chunk_size=4096):
    return _reduced_statistics(hidden, weight, bias, selected, chunk_size, None)[0]


@dataclass
class PrunedOTTCOutput:
    loss: torch.Tensor
    vocabulary: BatchVocabulary
    frame_indices: torch.Tensor
    target_indices: torch.Tensor
    transport_weights: torch.Tensor


def _check_mass(mass, mask, name):
    if mass.shape != mask.shape or not mass.is_floating_point():
        raise ValueError(f'{name} must match its mask and be floating point')
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
        raise ValueError(f'{name} must be finite and nonnegative')
    if bool((mass[~mask] != 0).any()):
        raise ValueError(f'{name} must be zero at padded positions')
    if not torch.allclose(mass.sum(-1), torch.ones_like(mass.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError(f'{name} must sum to one per utterance')


def pruned_ottc_loss(hidden, weight, bias, targets, frame_mass, *, input_mask=None,
                     target_mass=None, cost='jsd', chunk_size=4096, blank_id=0,
                     jsd_function=None):
    """Sum original OTTC costs with batch-local IDs and unchanged CDF transport.

    Inputs are encoder states (N,T,D), a full linear head (V,D), padded global
    target IDs (N,U), and normalized per-frame masses (N,T). Padding is -100;
    blank never enters the target vocabulary. Only first derivatives are supported.
    Pass the frozen upstream per-class JSD function, or let it load lazily.
    """
    if cost not in ('ce', 'jsd'):
        raise ValueError('cost must be ce or original upstream jsd')
    if hidden.ndim != 3 or targets.shape[0] != hidden.shape[0]:
        raise ValueError('hidden must have shape (N,T,D) with one target row per utterance')
    for value in (targets, frame_mass, input_mask, target_mass):
        if value is not None and value.device != hidden.device:
            raise ValueError('all OTTC inputs must share a device')
    vocabulary = remap_targets(targets, len(weight), blank_id=blank_id)
    if input_mask is None:
        input_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
    else:
        input_mask = input_mask.bool()
    if input_mask.shape != hidden.shape[:2] or not bool(input_mask.any(-1).all()):
        raise ValueError('input_mask must select at least one frame per utterance')
    if bool(((~input_mask[:, :-1]) & input_mask[:, 1:]).any()):
        raise ValueError('frames must be right-padded')
    valid_targets = targets != -100
    if target_mass is None:
        target_mass = valid_targets.to(frame_mass.dtype) / vocabulary.lengths[:, None]
    _check_mass(frame_mass, input_mask, 'frame_mass')
    _check_mass(target_mass, valid_targets, 'target_mass')
    if cost == 'jsd' and jsd_function is None:
        from reproduction.zapformer_s3t2_native_20261012.archived_ot_loss import jensen_shannon_divergence
        jsd_function = jensen_shannon_divergence
    cumulative_frames = frame_mass.cumsum(-1)
    cumulative_targets = target_mass.cumsum(-1)
    edges = torch.cat((cumulative_frames, cumulative_targets), -1).sort(-1).values
    transport = torch.diff(functional.pad(edges, (1, 0)), dim=-1)
    frame_indices = torch.searchsorted(cumulative_frames.contiguous(), edges.contiguous(), right=False)
    target_indices = torch.searchsorted(cumulative_targets.contiguous(), edges.contiguous(), right=False)
    frame_indices = torch.minimum(frame_indices, input_mask.sum(-1)[:, None] - 1)
    target_indices = torch.minimum(target_indices, vocabulary.lengths[:, None] - 1)
    batch_indices, slots = (transport > 0).nonzero(as_tuple=True)
    frame_offsets = input_mask.sum(-1).cumsum(0) - input_mask.sum(-1)
    paired_frames = frame_offsets[batch_indices] + frame_indices[batch_indices, slots]
    paired_targets = vocabulary.targets[batch_indices, target_indices[batch_indices, slots]]
    log_probs, background = _reduced_statistics(hidden[input_mask], weight, bias,
        vocabulary.selected, chunk_size, jsd_function if cost == 'jsd' else None)
    if cost == 'ce':
        pair_costs = -log_probs[paired_frames, paired_targets]
    else:
        probabilities = log_probs.exp()
        correction = (jsd_function(probabilities, torch.ones_like(probabilities, dtype=torch.float32))
                      - jsd_function(probabilities, torch.zeros_like(probabilities, dtype=torch.float32)))
        pair_costs = background[paired_frames] + correction[paired_frames, paired_targets]
    loss = (transport[batch_indices, slots] * pair_costs).sum()
    return PrunedOTTCOutput(loss, vocabulary, frame_indices, target_indices, transport)