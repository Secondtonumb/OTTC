"""Local CIF-inspired refinement of supplied-phone forced-alignment boundaries.

Original implementation of cumulative mass crossing, inspired by:
https://github.com/MingLunHan/CIF-PyTorch/blob/main/modules/cif_middleware.py

The bins below parameterize a learned density, NOT higher-resolution acoustic
observations. No original CIF code, quantity objective, or tail firing is used.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def mass_quantiles(mass: Tensor, edges: Tensor, quantiles: Tensor) -> Tensor:
    """Inverse CDF with uniform density inside each cell, in edges' units.

    Shapes: mass [..., K], edges [..., K+1], quantiles [..., Q]. Leading
    dimensions must match. Nonnegative mass need not sum to one. Zero cells
    are skipped; each row must have positive total mass. Arbitrarily many
    quantiles may cross the same cell (no single-fire-per-frame limitation).
    Piecewise differentiable in positive mass and edges, including the
    fraction of the crossing cell. Selection of a cell is discrete.
    """
    if not mass.is_floating_point():
        raise ValueError("mass must be floating point")
    dtype = torch.float64 if mass.dtype == torch.float64 else torch.float32
    mass, edges, quantiles = (x.to(dtype=dtype) for x in (mass, edges, quantiles))
    if mass.ndim < 1 or mass.shape[-1] == 0:
        raise ValueError("mass needs at least one cell")
    if edges.shape != (*mass.shape[:-1], mass.shape[-1] + 1):
        raise ValueError("edges shape must match mass with one extra endpoint")
    if quantiles.shape[:-1] != mass.shape[:-1]:
        raise ValueError("quantiles and mass must have matching leading dimensions")
    if not all(torch.isfinite(x).all() for x in (mass, edges, quantiles)):
        raise ValueError("inputs must be finite")
    if (mass < 0).any() or (edges.diff() < 0).any():
        raise ValueError("mass must be nonnegative and edges nondecreasing")
    total = mass.sum(-1, keepdim=True)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("each row needs finite positive total mass")
    if ((quantiles < 0) | (quantiles > 1)).any():
        raise ValueError("quantiles must lie in [0, 1]")
    cdf = torch.cat([torch.zeros_like(total), (mass / total).cumsum(-1)], -1)
    # Pin the endpoint, including FP roundoff. First >= crossing gives the
    # leftmost inverse when a quantile lands exactly on a zero-mass plateau.
    cdf = torch.cat([cdf[..., :-1], torch.ones_like(total)], -1)
    cell = torch.searchsorted(cdf[..., 1:].contiguous(), quantiles.contiguous())
    cell = cell.clamp(max=mass.shape[-1] - 1)
    before, after = cdf.gather(-1, cell), cdf.gather(-1, cell + 1)
    left, right = edges.gather(-1, cell), edges.gather(-1, cell + 1)
    fraction = (quantiles - before) / (after - before).clamp_min(
        torch.finfo(dtype).tiny
    )
    value = left + fraction.clamp(0, 1) * (right - left)
    value = torch.where(quantiles == 0, edges[..., :1], value)
    return torch.where(quantiles == 1, edges[..., -1:], value)


@dataclass
class BoundaryRefinement:
    times: Tensor
    coarse_times: Tensor
    offsets: Tensor
    window_edges: Tensor
    mass: Tensor


class CIFBoundaryRefiner(nn.Module):
    """One utterance at a time; slice padding before calling.

    Inputs contain no reference timestamps. `phone_ids` identifies token
    POSITIONS, including repeated phones. `coarse_times` has N-1 internal
    boundaries for N supplied phones. All times are seconds.

    Default acoustic clock is duration / T, matching native-k2. Optional
    frame_edges specifies a different clock explicitly. Acoustic features are
    linearly interpolated between frame-cell centers, with endpoint clamping.
    Symmetric, nonoverlapping windows guarantee identity at initialization
    and prevent boundary inversions without sorting or dropping tokens.
    """

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        hidden_dim: int = 128,
        phone_dim: int = 32,
        bins: int = 8,
        max_shift_ms: float = 20.0,
    ):
        super().__init__()
        if min(feature_dim, vocab_size, hidden_dim, phone_dim) < 1:
            raise ValueError("dimensions must be positive")
        if bins < 2 or bins % 2:
            raise ValueError(
                "use an even number of bins >= 2 for identity initialization"
            )
        if not math.isfinite(max_shift_ms) or max_shift_ms <= 0:
            raise ValueError("max_shift_ms must be finite and positive")
        self.config = dict(
            feature_dim=feature_dim,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            phone_dim=phone_dim,
            bins=bins,
            max_shift_ms=max_shift_ms,
        )
        self.phone_embedding = nn.Embedding(vocab_size, phone_dim)
        self.acoustic = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU()
        )
        self.density = nn.Sequential(
            nn.Linear(hidden_dim + 2 * phone_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.density[-1].weight)
        nn.init.zeros_(self.density[-1].bias)

    def forward(
        self,
        features: Tensor,
        coarse_times: Tensor,
        phone_ids: Tensor,
        duration: float,
        *,
        frame_edges: Tensor | None = None
    ) -> BoundaryRefinement:
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != self.config["feature_dim"]
        ):
            raise ValueError("features must be nonempty [valid_frames, feature_dim]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must be finite floating point")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        # Keep timestamp/CDF arithmetic out of mixed precision.
        coarse = coarse_times.to(device=features.device, dtype=torch.float32)
        ids = phone_ids.to(device=features.device)
        if coarse.ndim != 1 or ids.ndim != 1 or ids.numel() != coarse.numel() + 1:
            raise ValueError("N phones require N-1 internal coarse boundaries")
        if (
            ids.dtype != torch.long
            or (ids < 0).any()
            or (ids >= self.config["vocab_size"]).any()
        ):
            raise ValueError("phone_ids must be long integers within the vocabulary")
        if (
            not torch.isfinite(coarse).all()
            or (coarse < 0).any()
            or (coarse > duration).any()
            or (coarse.diff() < 0).any()
        ):
            raise ValueError(
                "coarse boundaries must be finite, ordered and inside the utterance"
            )
        if frame_edges is None:
            frame_edges = torch.linspace(
                0, duration, len(features) + 1, device=features.device
            )
        else:
            frame_edges = frame_edges.to(device=features.device, dtype=torch.float32)
            if (
                frame_edges.shape != (len(features) + 1,)
                or not torch.isfinite(frame_edges).all()
                or (frame_edges.diff() <= 0).any()
            ):
                raise ValueError(
                    "frame_edges must contain T+1 finite, strictly increasing times"
                )
        bins = self.config["bins"]
        if not coarse.numel():
            return BoundaryRefinement(
                coarse,
                coarse,
                coarse,
                coarse.new_empty(0, bins + 1),
                coarse.new_empty(0, bins),
            )
        previous = torch.cat([coarse.new_zeros(1), coarse[:-1]])
        following = torch.cat([coarse[1:], coarse.new_tensor([duration])])
        radius = torch.minimum((coarse - previous) / 2, (following - coarse) / 2)
        radius = radius.clamp(max=self.config["max_shift_ms"] / 1000)
        relative_edges = torch.linspace(-1, 1, bins + 1, device=features.device)
        edges = coarse[:, None] + radius[:, None] * relative_edges
        sample_times = (edges[:, :-1] + edges[:, 1:]) / 2
        centers = (frame_edges[:-1] + frame_edges[1:]) / 2
        # Even a one-frame utterance is supported; it has no extra observations.
        right = torch.searchsorted(centers, sample_times.contiguous()).clamp(
            max=len(features) - 1
        )
        left = (right - 1).clamp(min=0)
        fraction = (
            (sample_times - centers[left])
            / (centers[right] - centers[left]).clamp_min(1e-12)
        ).clamp(0, 1)
        projected = self.acoustic(features)
        sampled = projected[left] + fraction[..., None].to(projected.dtype) * (
            projected[right] - projected[left]
        )
        pair = torch.cat(
            [self.phone_embedding(ids[:-1]), self.phone_embedding(ids[1:])], -1
        )
        relative = (relative_edges[:-1] + relative_edges[1:]) / 2
        context = torch.cat(
            [
                sampled,
                pair[:, None, :].expand(-1, bins, -1),
                relative[None, :, None].expand(len(coarse), -1, -1).to(sampled.dtype),
            ],
            -1,
        )
        mass = self.density(context).squeeze(-1).float().softmax(-1)
        position = mass_quantiles(
            mass, edges, mass.new_full((len(coarse), 1), 0.5)
        ).squeeze(-1)
        return BoundaryRefinement(position, coarse, position - coarse, edges, mass)


def boundary_loss(
    result: BoundaryRefinement, reference_times: Tensor, *, beta_ms: float = 5.0
) -> Tensor:
    """Smooth-L1 in milliseconds; no CTC, no reference-based inference clipping.

    Gold boundaries outside the reachable window remain in the loss/metrics.
    An empty utterance contributes a differentiable zero when possible.
    """
    reference = reference_times.to(result.times)
    if reference.shape != result.times.shape or not torch.isfinite(reference).all():
        raise ValueError("reference must contain one finite time per internal boundary")
    if not math.isfinite(beta_ms) or beta_ms <= 0:
        raise ValueError("beta_ms must be positive")
    if not reference.numel():
        return result.times.sum() * 0
    return F.smooth_l1_loss(result.times * 1000, reference * 1000, beta=beta_ms)
