"""Training-free refinement using the frozen OTTC alignment branch's mass."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .cif_boundary import mass_quantiles


@dataclass
class FrozenMassRefinement:
    times: Tensor
    coarse_times: Tensor
    offsets: Tensor
    global_times: Tensor
    accepted: Tensor
    window_edges: Tensor


def refine_from_ottc_mass(
    frame_mass: Tensor,
    coarse_times: Tensor,
    duration: float,
    *,
    max_shift_ms: float = 20.0,
    frame_edges: Tensor | None = None,
    target_mass: Tensor | None = None,
) -> FrozenMassRefinement:
    """Reuse learned alpha without training or reference boundary timestamps.

    For original single-head OTTC, N supplied targets have beta_j=1/N.
    Crossings of cumsum(alpha) at j/N yield global continuous candidates.
    Accept candidate j only inside a symmetric window around K2 boundary j;
    otherwise return that original K2 boundary. Neighbor midpoint constraints
    guarantee order even when only some candidates are accepted. No clipping
    a distant crossing onto a local window and no moving its fractional part
    to an unrelated K2 frame. `accepted` reports this consistency gate, NOT
    confidence or accuracy. It is not an acoustically calibrated probability.

    The frame-cell uniform-density assumption is the same interpolation used
    by CIF mass splitting. This does not add sub-frame acoustic observations.
    Supply only valid (unpadded) frame masses from the model's correct branch;
    regfrozen checkpoints require wav2vec2_freeze -> lm_weight, not the current
    prediction encoder. N comes from coarse_times length + 1, not timestamps.
    """
    if frame_mass.ndim != 1 or not frame_mass.numel():
        raise ValueError("frame_mass must be a nonempty valid-frame vector")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(max_shift_ms) or max_shift_ms <= 0:
        raise ValueError("max_shift_ms must be finite and positive")
    dtype = torch.float64 if frame_mass.dtype == torch.float64 else torch.float32
    coarse = coarse_times.to(device=frame_mass.device, dtype=dtype)
    if (
        coarse.ndim != 1
        or not torch.isfinite(coarse).all()
        or (coarse < 0).any()
        or (coarse > duration).any()
        or (coarse.diff() < 0).any()
    ):
        raise ValueError(
            "coarse boundaries must be finite, ordered and within duration"
        )
    n = len(coarse) + 1
    if target_mass is None:
        thresholds = torch.arange(1, n, device=frame_mass.device, dtype=dtype) / n
    else:
        beta = target_mass.to(device=frame_mass.device, dtype=dtype)
        if (
            beta.shape != (n,)
            or not torch.isfinite(beta).all()
            or (beta <= 0).any()
            or not torch.isfinite(beta.sum())
        ):
            raise ValueError(
                "target_mass needs N finite positive masses with finite sum"
            )
        thresholds = (beta / beta.sum()).cumsum(0)[:-1]
    if frame_edges is None:
        edges = torch.linspace(
            0, duration, len(frame_mass) + 1, device=frame_mass.device, dtype=dtype
        )
    else:
        edges = frame_edges.to(device=frame_mass.device, dtype=dtype)
        if edges.shape != (len(frame_mass) + 1,) or (edges.diff() <= 0).any():
            raise ValueError("frame_edges must contain T+1 strictly increasing times")
    candidate = mass_quantiles(frame_mass, edges, thresholds)
    if not len(coarse):
        return FrozenMassRefinement(
            coarse,
            coarse,
            coarse,
            candidate,
            torch.empty(0, dtype=torch.bool, device=coarse.device),
            coarse.new_empty(0, 2),
        )
    previous = torch.cat([coarse.new_zeros(1), coarse[:-1]])
    following = torch.cat([coarse[1:], coarse.new_tensor([duration])])
    radius = torch.minimum((coarse - previous) / 2, (following - coarse) / 2).clamp(
        max=max_shift_ms / 1000
    )
    windows = torch.stack([coarse - radius, coarse + radius], -1)
    accepted = (candidate >= windows[:, 0]) & (candidate <= windows[:, 1])
    times = torch.where(accepted, candidate, coarse)
    return FrozenMassRefinement(
        times, coarse, times - coarse, candidate, accepted, windows
    )
