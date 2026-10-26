"""Experimental local GW penalties on occurrence-level transport plans.

ASOT-inspired structure (mingu6/action_seg_ot, src/asot.py), independently
implemented with prefix sums, physical-time windows and a consistent objective.
No generic GW metric or convergence claim: ASOT uses a finite unrolled solver.
All production arithmetic is FP32. Double is retained only for gradcheck.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GWOptions:
    mode: str = 'monotone'
    weight: float = 0.1
    window_frames: int = 2
    iterations: int = 25
    step_size: float = 0.1
    entropy: float = 0.05
    column_kl: float = 1.0
    order_prior: float = 0.1

    def __post_init__(self):
        if self.mode not in ('monotone', 'asot'):
            raise ValueError('mode must be monotone or asot')
        for name in ('weight', 'entropy', 'column_kl', 'order_prior'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name)
        if not math.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError('step_size')
        if self.window_frames < 1 or self.iterations < 1:
            raise ValueError('positive window and iteration count required')


def working(x):
    return x if x.dtype == torch.float64 else x.float()


def neighborhood_sum(x, radius):
    """Sum valid temporal neighbors, excluding self; O(NK), no NxN matrix."""
    n = x.shape[0]
    idx = torch.arange(n, device=x.device)
    prefix = torch.cat((torch.zeros_like(x[:1]), x.cumsum(0)), dim=0)
    return prefix[(idx + radius + 1).clamp(max=n)] - prefix[(idx - radius).clamp(min=0)] - x


def structural_gradient(q, radius):
    """Gradient of R=.5*N^2/pairs * <Q,A Q (11'-I)>.

    Target axis is occurrence index, NOT phone identity. N^2/pairs makes R
    .5 times the local disagreement probability for uniform frame marginals.
    Learned marginals remain part of Q: this is intentionally a plan penalty,
    not a posterior loss. It can be gamed by mass concentration; log that.
    """
    n = q.shape[0]
    if n < 2:
        return q * 0
    r = min(radius, n - 1)
    pairs = 2 * r * n - r * (r + 1)
    return neighborhood_sum(q.sum(-1, keepdim=True) - q, r) * (n * n / pairs)


def structure(q, radius):
    return 0.5 * (q * structural_gradient(q, radius)).sum()


def monotone_plan(a, b):
    """Interval intersection form of the existing cumulative-mass transport."""
    a, b = working(a), working(b)
    ca, cb = a.cumsum(0), b.cumsum(0)
    la = torch.cat((torch.zeros_like(ca[:1]), ca[:-1]))
    lb = torch.cat((torch.zeros_like(cb[:1]), cb[:-1]))
    return (torch.minimum(ca[:, None], cb[None, :]) -
            torch.maximum(la[:, None], lb[None, :])).clamp_min(0)


def ag_cost(logits, labels):
    """Reuse the original upstream AG function, including its target epsilon."""
    from ottc.ops_loss.ot_loss import jensen_shannon_divergence
    logits = working(logits)
    p = logits.softmax(-1)
    target = F.one_hot(labels, logits.shape[-1]).to(p.dtype)
    return jensen_shannon_divergence(p[:, None, :], target[None, :, :]).sum(-1)


def positional_cost(n, k, device, dtype):
    # Soft prior only. Never claim hard monotonicity for this solver.
    t = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    u = (torch.arange(k, device=device, dtype=dtype) + 0.5) / k
    return (t[:, None] - u[None, :]).abs()


def objective_terms(q, cost, b, options):
    tiny = torch.finfo(q.dtype).tiny
    col = q.sum(0)
    linear = (q * cost).sum()
    gw = structure(q, options.window_frames)
    # Generalized KL. Unit total mass makes the last two terms cancel.
    kl = (col * (col.clamp_min(tiny).log() - b.log()) - col + b).sum()
    neg_entropy = (q * (q.clamp_min(tiny).log() - 1)).sum()
    prior = (q * positional_cost(*q.shape, q.device, q.dtype)).sum()
    total = linear + options.weight * gw + options.column_kl * kl
    total = total + options.entropy * neg_entropy + options.order_prior * prior
    return dict(total=total, ag=linear, gw=gw, column_kl=kl,
                negative_entropy=neg_entropy, order_prior=prior)


def asot_plan(cost, a, b, options):
    """Unrolled log-domain mirror descent; exact row=a, soft column~b.

    Unlike the upstream detached pseudo-label training, gradients propagate
    through every iteration into both AG logits and the learned frame masses.
    No in-place updates, .item() step heuristics or silent NaN replacement.
    """
    cost, a, b = working(cost), working(a), working(b)
    tiny = torch.finfo(cost.dtype).tiny
    loga, logb = a.clamp_min(tiny).log(), b.log()
    logq = loga[:, None] + logb[None, :]
    prior = positional_cost(*cost.shape, cost.device, cost.dtype)
    for _ in range(options.iterations):
        q = logq.exp()
        grad = cost + options.weight * structural_gradient(q, options.window_frames)
        grad = grad + options.column_kl * (q.sum(0).clamp_min(tiny).log() - logb)[None, :]
        grad = grad + options.entropy * logq + options.order_prior * prior
        logq = loga[:, None] + F.log_softmax(logq - options.step_size * grad, dim=-1)
    return logq.exp()


@torch.no_grad()
def plan_diagnostics(q, a, b):
    n, k = q.shape
    chosen = q.argmax(-1)
    return dict(row_error=(q.sum(-1) - a).abs().max(),
                column_l1=(q.sum(0) - b).abs().sum(),
                backward_fraction=(chosen[1:] < chosen[:-1]).float().mean() if n > 1 else q.new_zeros(()),
                occurrence_coverage=chosen.unique().numel() / k,
                effective_frame_fraction=1 / (n * a.square().sum()),
                max_frame_mass=a.max())
