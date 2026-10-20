# OTTC implementation adapted from IF-MDD's MIT-licensed ot_loss.py.
# SPDX-FileCopyrightText: 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: MIT
"""Loss functions adapted from IF-MDD for ZipA CTC training."""

import math
from dataclasses import dataclass
from typing import Tuple

import k2
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

SUBPHONETIC_TOPOLOGY_NUM_STATES = {
    "s1_t1": 1,
    "s2_t1": 2,
    "s2_t1_star": 2,
    "s3_t2": 3,
    "s3_t2_star": 3,
    "s3_t2_star_star": 3,
}


def batched_ottc_loss_bucketized(
    logits: Tensor,
    targets: Tensor,
    frame_weights: Tensor,
    target_weights: Tensor,
    frame_mask: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Compute IF-MDD's ordered 1-D optimal-transport cross-entropy loss."""
    device = logits.device
    frame_cdf = torch.cumsum(frame_weights, dim=1)
    target_cdf = torch.cumsum(target_weights, dim=1)
    merged_cdf = torch.sort(torch.cat([frame_cdf, target_cdf], dim=1), dim=1).values

    frame_bins = frame_mask.sum(dim=1, keepdim=True).to(torch.int64)
    target_bins = target_mask.sum(dim=1, keepdim=True).to(torch.int64)
    frame_indices = torch.vmap(torch.bucketize)(merged_cdf, frame_cdf)
    target_indices = torch.vmap(torch.bucketize)(merged_cdf, target_cdf)
    frame_indices = torch.minimum(frame_indices, frame_bins - 1)
    target_indices = torch.minimum(target_indices, target_bins - 1)

    batch_offsets = torch.arange(logits.size(0), device=device).unsqueeze(1)
    flat_frame_indices = (frame_indices + batch_offsets * logits.size(1)).flatten()
    flat_target_indices = (target_indices + batch_offsets * targets.size(1)).flatten()
    transport_weights = torch.diff(
        F.pad(merged_cdf, (1, 0), value=0.0), dim=1
    ).flatten()
    positive = transport_weights > 0

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1, targets.size(-1))
    costs = F.cross_entropy(
        flat_logits[flat_frame_indices[positive]],
        flat_targets[flat_target_indices[positive]].float(),
        reduction="none",
    )
    return (transport_weights[positive] * costs).sum()


def batched_ordered_multilabel_loss(
    logits: Tensor,
    targets: Tensor,
    frame_weights: Tensor,
    target_weights: Tensor,
    frame_mask: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Ordered-transport BCE for phone-attached diacritic labels."""

    frame_cdf = torch.cumsum(frame_weights, dim=1)
    target_cdf = torch.cumsum(target_weights, dim=1)
    merged_cdf = torch.sort(
        torch.cat([frame_cdf, target_cdf], dim=1), dim=1
    ).values

    frame_bins = frame_mask.sum(dim=1, keepdim=True).to(torch.int64)
    target_bins = target_mask.sum(dim=1, keepdim=True).to(torch.int64)
    frame_indices = torch.vmap(torch.bucketize)(merged_cdf, frame_cdf)
    target_indices = torch.vmap(torch.bucketize)(merged_cdf, target_cdf)
    frame_indices = torch.minimum(frame_indices, frame_bins - 1)
    target_indices = torch.minimum(target_indices, target_bins - 1)

    batch_offsets = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
    flat_frame_indices = (
        frame_indices + batch_offsets * logits.size(1)
    ).flatten()
    flat_target_indices = (
        target_indices + batch_offsets * targets.size(1)
    ).flatten()
    transport_weights = torch.diff(
        F.pad(merged_cdf, (1, 0), value=0.0), dim=1
    ).flatten()
    positive = transport_weights > 0

    frame_logits = logits.reshape(-1, logits.size(-1))[
        flat_frame_indices[positive]
    ]
    phone_targets = targets.reshape(-1, targets.size(-1))[
        flat_target_indices[positive]
    ].to(frame_logits.dtype)
    costs = F.binary_cross_entropy_with_logits(
        frame_logits, phone_targets, reduction="none"
    ).mean(dim=-1)
    return (transport_weights[positive] * costs).sum()


def make_ottc_inputs(
    logits: Tensor,
    input_lengths: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    weight_logits: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build normalized frame/label measures for ordered transport."""
    device = logits.device
    frame_steps = torch.arange(logits.size(1), device=device).unsqueeze(0)
    frame_mask = frame_steps < input_lengths.to(device).unsqueeze(1)
    frame_weights = F.softmax(
        weight_logits.masked_fill(~frame_mask, -torch.inf), dim=1
    )

    target_steps = torch.arange(targets.size(1), device=device).unsqueeze(0)
    target_mask = target_steps < target_lengths.to(device).unsqueeze(1)
    target_weights = target_mask / target_mask.sum(dim=1, keepdim=True).clamp_min(1)
    one_hot_targets = F.one_hot(
        targets.to(device).long(), num_classes=logits.size(-1)
    )
    return frame_weights, target_weights, frame_mask, one_hot_targets


def pad_flat_targets(targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Convert concatenated CTC targets to a zero-padded batch."""
    sequences = []
    offset = 0
    for length in target_lengths.tolist():
        sequences.append(targets[offset : offset + length])
        offset += length
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)


def expand_subphonetic_targets(
    targets: Tensor,
    target_lengths: Tensor,
    num_states: int = 3,
) -> Tuple[Tensor, Tensor]:
    """Expand token IDs into ordered S0/S1/S2 IDs.

    Blank remains ID 0. For a non-blank token ``p``, state ``s`` is encoded as
    ``1 + (p - 1) * num_states + s``.
    """
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}")
    sequences = []
    offset = 0
    for length in target_lengths.tolist():
        token_ids = targets[offset : offset + length].long()
        offset += length
        if torch.any(token_ids <= 0):
            raise ValueError("Subphonetic targets must not contain blank/non-positive IDs")
        base = 1 + (token_ids - 1) * num_states
        states = torch.arange(num_states, device=targets.device)
        sequences.append((base.unsqueeze(1) + states).reshape(-1))
    expanded_lengths = target_lengths * num_states
    if not sequences:
        return targets.new_empty((0,), dtype=torch.long), expanded_lengths
    return torch.cat(sequences), expanded_lengths


@dataclass(frozen=True)
class TopologyArc:
    src: int
    dst: int
    label: int


def add_subphonetic_phone_arcs(
    arcs: list[TopologyArc],
    src: int,
    dst: int,
    state_ids: tuple[int, ...],
    topology: str,
    first_node: int,
) -> int:
    """Add the six paper-figure IF-MDD topology variants for one phone."""
    if topology not in SUBPHONETIC_TOPOLOGY_NUM_STATES:
        raise ValueError(f"Unsupported topology: {topology}")
    expected_states = SUBPHONETIC_TOPOLOGY_NUM_STATES[topology]
    if len(state_ids) != expected_states:
        raise ValueError(
            f"{topology} requires {expected_states} states, got {len(state_ids)}"
        )

    if topology == "s1_t1":
        (s0,) = state_ids
        n0 = first_node
        arcs.extend(
            [
                TopologyArc(src, n0, s0),
                TopologyArc(n0, n0, s0),
                TopologyArc(n0, dst, 0),
            ]
        )
        return first_node + 1

    if topology in {"s2_t1", "s2_t1_star"}:
        s0, s1 = state_ids
        n0, n1 = first_node, first_node + 1
        arcs.extend(
            [
                TopologyArc(src, dst, s0),
                TopologyArc(src, n0, s0),
                TopologyArc(n0, dst, s1),
                TopologyArc(n0, n1, s1),
                TopologyArc(n1, n1, s1),
                TopologyArc(n1, dst, s1),
            ]
        )
        if topology == "s2_t1_star":
            arcs.append(TopologyArc(n0, n0, s0))
        return first_node + 2

    s0, s1, s2 = state_ids
    n0, n1, n2 = first_node, first_node + 1, first_node + 2
    arcs.extend(
        [
            TopologyArc(src, n0, s0),
            TopologyArc(n0, dst, s2),
            TopologyArc(n0, n1, s1),
            TopologyArc(n1, n1, s1),
            TopologyArc(n1, dst, s2),
        ]
    )
    if topology in {"s3_t2_star", "s3_t2_star_star"}:
        arcs.extend(
            [
                TopologyArc(n0, n2, s2),
                TopologyArc(n1, n2, s2),
                TopologyArc(n2, n2, s2),
                TopologyArc(n2, dst, s2),
            ]
        )
    if topology == "s3_t2_star_star":
        arcs.append(TopologyArc(n0, n0, s0))
    return first_node + 3


def build_subphonetic_utterance_arcs(
    token_ids: list[int],
    topology: str,
    blank_id: int = 0,
) -> tuple[list[TopologyArc], int]:
    """Build a transcript-conditioned acceptor matching IF-MDD."""
    if topology not in SUBPHONETIC_TOPOLOGY_NUM_STATES:
        raise ValueError(f"Unsupported topology: {topology}")
    num_states = SUBPHONETIC_TOPOLOGY_NUM_STATES[topology]
    arcs: list[TopologyArc] = []
    boundary = 0
    next_free = 1
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id <= 0:
            continue
        arcs.append(TopologyArc(boundary, boundary, int(blank_id)))
        base = 1 + (token_id - 1) * num_states
        next_boundary = next_free + num_states
        next_free = add_subphonetic_phone_arcs(
            arcs,
            src=boundary,
            dst=next_boundary,
            state_ids=tuple(base + state for state in range(num_states)),
            topology=topology,
            first_node=next_free,
        )
        # Keep boundary IDs disjoint from the phone-internal topology nodes.
        boundary = next_boundary
        next_free = boundary + 1
    arcs.append(TopologyArc(boundary, boundary, int(blank_id)))
    return arcs, boundary


def build_s3t2_utterance_arcs(
    token_ids: list[int],
    topology: str,
    blank_id: int = 0,
) -> tuple[list[TopologyArc], int]:
    """Backward-compatible alias retained for existing S3 tests/callers."""
    return build_subphonetic_utterance_arcs(token_ids, topology, blank_id)


def _subphonetic_fsa_text(
    token_ids: list[int],
    topology: str,
    blank_id: int,
) -> str:
    arcs, final_state = build_subphonetic_utterance_arcs(
        token_ids, topology=topology, blank_id=blank_id
    )
    lines = [
        f"{arc.src} {arc.dst} {arc.label} 0.0"
        for arc in sorted(arcs, key=lambda item: (item.src, item.dst, item.label))
    ]
    super_final = final_state + 1
    lines.append(f"{final_state} {super_final} -1 0.0")
    lines.append(str(super_final))
    return "\n".join(lines) + "\n"


def topology_forward_sum_loss(
    log_probs: Tensor,
    input_lengths: Tensor,
    targets: Tensor,
    target_lengths: Tensor,
    topology: str,
    blank_id: int = 0,
    output_beam: float = 10.0,
) -> Tensor:
    """Exact forward-sum loss over transcript-conditioned S3T2 arcs.

    The returned value is a sum of per-utterance frame-normalized losses, so
    its scale is comparable to the existing OTTC sum (one distributional
    cross-entropy per utterance).
    """
    sequences = []
    offset = 0
    for length in target_lengths.detach().cpu().tolist():
        sequences.append(
            [int(value) for value in targets[offset : offset + length].detach().cpu()]
        )
        offset += length

    order = torch.argsort(input_lengths, descending=True)
    graph_device = log_probs.device
    graphs = [
        k2.arc_sort(
            k2.Fsa.from_str(
                _subphonetic_fsa_text(
                    sequences[int(old_idx)],
                    topology=topology,
                    blank_id=blank_id,
                ),
                acceptor=True,
            ).to(graph_device)
        )
        for old_idx in order.detach().cpu().tolist()
    ]
    supervision = torch.tensor(
        [
            [new_idx, 0, int(input_lengths[int(old_idx)].item())]
            for new_idx, old_idx in enumerate(order.detach().cpu().tolist())
        ],
        dtype=torch.int32,
        device="cpu",
    )
    graph_vec = k2.arc_sort(k2.Fsa.from_fsas(graphs).to(graph_device))
    dense = k2.DenseFsaVec(log_probs[order].float(), supervision)
    per_loss = k2.ctc_loss(
        decoding_graph=graph_vec,
        dense_fsa_vec=dense,
        output_beam=float(output_beam),
        reduction="none",
        use_double_scores=True,
    )
    sorted_lengths = input_lengths[order].to(
        device=per_loss.device, dtype=per_loss.dtype
    )
    return (per_loss / sorted_lengths.clamp_min(1)).sum().to(log_probs.device)


def build_s3t2_phone_loop(
    num_phone_tokens: int,
    topology: str,
    blank_id: int = 0,
) -> tuple[list[TopologyArc], dict[int, int], int]:
    """Build a free-phone-loop graph and entry-arc phone outputs."""
    arcs = [TopologyArc(0, 0, int(blank_id))]
    arc_outputs: dict[int, int] = {}
    next_free = 1
    for token_id in range(1, int(num_phone_tokens) + 1):
        base = 1 + (token_id - 1) * 3
        before = len(arcs)
        next_free = add_subphonetic_phone_arcs(
            arcs,
            src=0,
            dst=0,
            state_ids=(base, base + 1, base + 2),
            topology=topology,
            first_node=next_free,
        )
        for arc_index in range(before, len(arcs)):
            arc = arcs[arc_index]
            if arc.src == 0 and arc.label != blank_id:
                arc_outputs[arc_index] = token_id
    num_graph_states = 1 + max(max(arc.src, arc.dst) for arc in arcs)
    return arcs, arc_outputs, num_graph_states


def s3t2_topology_viterbi(
    log_probs: Tensor,
    input_lengths: Tensor,
    topology: str,
    blank_id: int = 0,
) -> list[list[int]]:
    """Decode the best legal path through a batched k2 phone-loop graph."""
    if (log_probs.shape[-1] - 1) % 3 != 0:
        raise ValueError(
            "S3T2 logits must contain blank + 3 rows per phone, got "
            f"{log_probs.shape[-1]} classes"
        )
    num_phone_tokens = (log_probs.shape[-1] - 1) // 3
    arcs, arc_outputs, num_states = build_s3t2_phone_loop(
        num_phone_tokens=num_phone_tokens,
        topology=topology,
        blank_id=blank_id,
    )
    super_final = num_states
    lines = []
    for arc_index, arc in enumerate(arcs):
        output = arc_outputs.get(arc_index, 0)
        lines.append(
            f"{arc.src} {arc.dst} {arc.label} {output} 0.0"
        )
    lines.append(f"0 {super_final} -1 -1 0.0")
    lines.append(str(super_final))
    graph_text = "\n".join(
        sorted(
            lines[:-1],
            key=lambda line: tuple(int(value) for value in line.split()[:3]),
        )
        + [lines[-1]]
    ) + "\n"

    order = torch.argsort(input_lengths, descending=True)
    ordered_lengths = input_lengths[order].detach().cpu()
    base_graph = k2.arc_sort(
        k2.Fsa.from_str(graph_text, acceptor=False).to(log_probs.device)
    )
    graph_vec = k2.Fsa.from_fsas(
        [base_graph for _ in range(log_probs.shape[0])]
    ).to(log_probs.device)
    supervision = torch.tensor(
        [
            [batch_index, 0, int(length)]
            for batch_index, length in enumerate(ordered_lengths.tolist())
        ],
        dtype=torch.int32,
        device="cpu",
    )
    dense = k2.DenseFsaVec(log_probs[order].detach().float(), supervision)
    lattice = k2.intersect_dense_pruned(
        graph_vec,
        dense,
        search_beam=20.0,
        output_beam=8.0,
        min_active_states=30,
        max_active_states=10000,
    )
    best_paths = k2.shortest_path(lattice, use_double_scores=True)
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        aux = best_paths.aux_labels.remove_values_leq(0)
        shape = best_paths.arcs.shape().compose(aux.shape)
        shape = shape.remove_axis(1).remove_axis(1)
        aux = k2.RaggedTensor(shape, aux.values)
    else:
        shape = best_paths.arcs.shape().remove_axis(1)
        aux = k2.RaggedTensor(shape, best_paths.aux_labels)
        aux = aux.remove_values_leq(0)
    ordered_hypotheses = aux.tolist()
    hypotheses = [[] for _ in ordered_hypotheses]
    for new_index, old_index in enumerate(order.detach().cpu().tolist()):
        hypotheses[old_index] = ordered_hypotheses[new_index]
    return hypotheses


def _s3t2_greedy_complete(
    frame_ids: Tensor,
    input_lengths: Tensor,
    *,
    num_states: int,
    blank_id: int,
) -> list[list[int]]:
    if num_states != 3:
        raise ValueError("S3T2 decoding requires exactly three states")
    hypotheses: list[list[int]] = []
    for row, length in zip(frame_ids.detach().cpu(), input_lengths.detach().cpu()):
        collapsed: list[int] = []
        previous = None
        for raw_id in row[: int(length)].tolist():
            state_id = int(raw_id)
            if state_id != previous and state_id != blank_id:
                collapsed.append(state_id)
            previous = state_id

        output: list[int] = []
        active_token = None
        stage = -1

        def finish_active() -> None:
            nonlocal active_token, stage
            if active_token is not None and stage == 2:
                output.append(active_token)
            active_token = None
            stage = -1

        for state_id in collapsed:
            token_id = (state_id - 1) // num_states + 1
            state = (state_id - 1) % num_states
            if token_id != active_token:
                finish_active()
                if state == 0:
                    active_token = token_id
                    stage = 0
                continue

            if state == stage:
                continue
            # Every S3T2 variant permits the two-emission completion S0->S2.
            # S1 is an optional middle state.  The *, ** variants differ only
            # in which topology substates have self-loops; consecutive greedy
            # repeats have already been collapsed above.
            if stage == 0 and state in (1, 2):
                stage = state
                continue
            if stage == 1 and state == 2:
                stage = 2
                continue
            if stage == 2 and state == 0:
                output.append(active_token)
                stage = 0
                continue

            finish_active()
            if state == 0:
                active_token = token_id
                stage = 0

        finish_active()
        hypotheses.append(output)
    return hypotheses


def subphonetic_greedy_complete(
    frame_ids: Tensor,
    input_lengths: Tensor,
    topology: str,
    blank_id: int = 0,
) -> list[list[int]]:
    """Collapse greedy substate paths for the six IF-MDD paper topologies."""
    if topology not in SUBPHONETIC_TOPOLOGY_NUM_STATES:
        raise ValueError(f"Unsupported topology: {topology}")
    num_states = SUBPHONETIC_TOPOLOGY_NUM_STATES[topology]
    if topology.startswith("s3_"):
        return _s3t2_greedy_complete(
            frame_ids,
            input_lengths,
            num_states=num_states,
            blank_id=blank_id,
        )

    hypotheses: list[list[int]] = []
    for row, length in zip(frame_ids.detach().cpu(), input_lengths.detach().cpu()):
        output: list[int] = []
        previous = None
        for raw_id in row[: int(length)].tolist():
            state_id = int(raw_id)
            if state_id == previous:
                continue
            previous = state_id
            if state_id == blank_id:
                continue
            state = (state_id - 1) % num_states
            # S1T1 and S2T1 emit the lexical phone on their S0 arc.
            if state == 0:
                output.append((state_id - 1) // num_states + 1)
        hypotheses.append(output)
    return hypotheses


def s3t2_greedy_complete(
    frame_ids: Tensor,
    input_lengths: Tensor,
    num_states: int = 3,
    blank_id: int = 0,
) -> list[list[int]]:
    """Collapse a completed S3T2 path: S0 -> [S1] -> S2."""
    return _s3t2_greedy_complete(
        frame_ids,
        input_lengths,
        num_states=num_states,
        blank_id=blank_id,
    )


def s3t2_star_star_greedy_complete(
    frame_ids: Tensor,
    input_lengths: Tensor,
    num_states: int = 3,
    blank_id: int = 0,
) -> list[list[int]]:
    """Collapse S3T2**; greedy phone completion is identical to S3T2."""
    return _s3t2_greedy_complete(
        frame_ids,
        input_lengths,
        num_states=num_states,
        blank_id=blank_id,
    )


class CTCLossWithLabelPriors(nn.Module):
    """k2 CTC with IF-MDD-style EMA acoustic label priors."""

    def __init__(
        self,
        vocab_size: int,
        blank: int = 0,
        prior_scaling_factor: float = 0.3,
        prior_momentum: float = 0.9,
        output_beam: float = 10.0,
    ) -> None:
        super().__init__()
        self.blank = blank
        self.prior_scaling_factor = prior_scaling_factor
        self.prior_momentum = prior_momentum
        self.output_beam = output_beam
        self.register_buffer("log_priors", torch.zeros(1, vocab_size), persistent=True)
        self.register_buffer("ema_log_priors", torch.zeros(1, vocab_size), persistent=True)
        self.register_buffer("prior_initialized", torch.tensor(False), persistent=True)
        self.register_buffer("log_priors_sum", torch.empty(0), persistent=False)
        self.register_buffer("num_prior_frames", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        collect_priors: bool,
    ) -> Tensor:
        """Accept `(N,T,C)` log probabilities and concatenated CTC targets."""
        if collect_priors:
            valid = [
                utterance[: int(length)]
                for utterance, length in zip(log_probs, input_lengths.tolist())
            ]
            flattened = torch.cat(valid, dim=0)
            batch_sum = torch.logsumexp(flattened, dim=0, keepdim=True).detach()
            if self.log_priors_sum.numel() == 0:
                self.log_priors_sum = batch_sum
            else:
                self.log_priors_sum = torch.logsumexp(
                    torch.stack([self.log_priors_sum, batch_sum], dim=-1), dim=-1
                )
            self.num_prior_frames += flattened.size(0)

        adjusted = log_probs
        if self.prior_initialized.item() and self.prior_scaling_factor > 0:
            adjusted = adjusted - self.prior_scaling_factor * self.log_priors

        padded_targets = pad_flat_targets(targets, target_lengths)
        supervision_segments, token_ids = self._encode_supervisions(
            padded_targets, target_lengths, input_lengths
        )
        graph = k2.ctc_graph(token_ids, modified=False, device=log_probs.device)
        dense_fsa = k2.DenseFsaVec(adjusted, supervision_segments)
        return k2.ctc_loss(
            decoding_graph=graph,
            dense_fsa_vec=dense_fsa,
            output_beam=self.output_beam,
            reduction="sum",
            use_double_scores=True,
        )

    @staticmethod
    def _encode_supervisions(targets, target_lengths, input_lengths):
        batch_size = targets.size(0)
        segments = torch.stack(
            [
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ],
            dim=1,
        ).to(torch.int32)
        indices = torch.argsort(segments[:, 2], descending=True)
        segments = segments[indices]
        lengths = target_lengths[indices].tolist()
        token_ids = [row[:length] for row, length in zip(targets[indices].tolist(), lengths)]
        return segments, token_ids

    @torch.no_grad()
    def finalize_epoch(self) -> bool:
        """Commit accumulated priors after one training epoch."""
        if self.log_priors_sum.numel() == 0 or self.num_prior_frames.item() == 0:
            return False

        counts = self.log_priors_sum.exp()
        frames = self.num_prior_frames.clone().to(counts.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(frames, op=dist.ReduceOp.SUM)
        new_log_priors = counts.clamp_min(torch.finfo(counts.dtype).tiny).log()
        new_log_priors -= math.log(float(frames.item()))

        if not self.prior_initialized.item():
            self.ema_log_priors = new_log_priors
            self.prior_initialized.fill_(True)
        else:
            self.ema_log_priors = (
                self.prior_momentum * self.ema_log_priors
                + (1.0 - self.prior_momentum) * new_log_priors
            )
        self.log_priors = self.ema_log_priors.clone()
        self.log_priors_sum = torch.empty(0, device=counts.device)
        self.num_prior_frames.zero_()
        return True
