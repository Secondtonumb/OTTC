"""Position-wise OTTC decoding with normalized transport-weighted evidence.

The caller supplies the token count: utterance-normalized alpha has no quantity
prediction. The default pools logits/log posterior; an optional probability
average is a separate decoding rule. No CTC collapse
is applied to the output positions, so adjacent equal tokens are preserved.
"""
from __future__ import annotations

from numbers import Integral
import torch


def uniform_transport(alpha: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Monotone 1D overlap plan [T, M], with uniform target mass 1/M.

    Alpha may be unnormalized. Zero-mass frames are retained with zero transport.
    Float64 cumulative sums avoid disappearing narrow bins at the end of a CDF.
    """
    if not isinstance(num_tokens, Integral) or isinstance(num_tokens, bool) or num_tokens < 0:
        raise ValueError("num_tokens must be a nonnegative integer")
    alpha = torch.as_tensor(alpha)
    if alpha.ndim != 1 or not torch.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError("alpha must be a finite nonnegative vector")
    if num_tokens == 0:
        return torch.zeros((alpha.numel(), 0), device=alpha.device, dtype=torch.float64)
    mass = alpha.double()
    total = mass.sum()
    if not total > 0:
        raise ValueError("positive token count requires positive alpha mass")
    mass = mass / total
    right = mass.cumsum(0)
    left = torch.cat((right.new_zeros(1), right[:-1]))
    edges = torch.arange(num_tokens + 1, device=alpha.device, dtype=torch.float64) / num_tokens
    return (torch.minimum(right[:, None], edges[None, 1:])
            - torch.maximum(left[:, None], edges[None, :-1])).clamp_min(0)


def pool_gamma_logits(logits: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Return [M, K] scores: (gamma / gamma.sum(0)).T @ logits.

    Log posterior is equivalent up to one class-independent constant per token.
    Zero-mass target columns are invalid; empty M is supported. Accumulate at
    least in FP32, including when model inference used FP16/BF16.
    """
    logits = torch.as_tensor(logits)
    if logits.ndim != 2 or not logits.is_floating_point() or logits.shape[1] == 0:
        raise ValueError("logits must be a floating [T, K] tensor with K > 0")
    gamma = torch.as_tensor(gamma, device=logits.device)
    if gamma.ndim != 2 or gamma.shape[0] != logits.shape[0]:
        raise ValueError("gamma must have shape [T, M] matching logits")
    if not torch.isfinite(gamma).all() or (gamma < 0).any():
        raise ValueError("gamma must be finite and nonnegative")
    g = gamma.double()
    beta = g.sum(0)
    if (beta <= 0).any():
        raise ValueError("every target position must receive positive mass")
    # Drop inactive rows before pooling: padding NaN/-inf must never contaminate
    # evidence via 0 * NaN. Active evidence is required to be finite.
    active = g.sum(1) > 0
    values = logits[active]
    if not torch.isfinite(values).all():
        raise ValueError("positive-mass frame evidence must be finite")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    weights = (g[active] / beta).to(dtype)
    return weights.T @ values.to(dtype)


def gamma_decode(logits: torch.Tensor, alpha: torch.Tensor, num_tokens: int,
                 valid_mask: torch.Tensor | None = None, *,
                 aggregation: str = "logits") -> torch.Tensor:
    """Predict one ID per position, without collapsing repeated output IDs.

    Supplying a reference-derived num_tokens is an oracle-length diagnostic,
    not text-free recognition. A valid mask removes padded frames before CDFs.
    """
    logits = torch.as_tensor(logits)
    if aggregation not in {"logits", "posterior"}:
        raise ValueError("aggregation must be logits or posterior")
    alpha = torch.as_tensor(alpha, device=logits.device)
    if logits.ndim != 2 or alpha.ndim != 1 or logits.shape[0] != alpha.numel():
        raise ValueError("expected logits [T, K] and alpha [T]")
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=logits.device, dtype=torch.bool)
        if mask.shape != alpha.shape:
            raise ValueError("valid_mask must match alpha")
        logits, alpha = logits[mask], alpha[mask]
    gamma = uniform_transport(alpha, num_tokens)
    # Softmax per frame, then weighted arithmetic average, is deliberately
    # distinct from averaging logits/log posterior. Both reuse the same gamma.
    evidence = logits.float().softmax(-1) if aggregation == "posterior" else logits
    return pool_gamma_logits(evidence, gamma).argmax(-1)


def greedy_token_count(frame_ids, ignore_ids=()) -> int:
    """Count greedy runs after dropping special runs, retaining blank splits."""
    previous = None
    count = 0
    ignored = set(ignore_ids)
    for value in frame_ids:
        token = int(value)
        if token != previous and token not in ignored:
            count += 1
        previous = token
    return count
