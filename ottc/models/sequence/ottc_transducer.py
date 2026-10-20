"""Blank-free autoregressive OTTC transducer, NOT the RNN-T likelihood.

Each target occurrence owns a separate uniform-mass bin. The original OTTC
transport selects (frame, occurrence) pairs; a Predictor and SpeechBrain Joint
make their phone costs history dependent. A separately supervised log-count
head supplies the bin count at inference. No target length enters recognize().
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from speechbrain.nnet.transducer.transducer_joint import Transducer_joint

from reproduction.zapformer_s3t2_native_20261012.archived_ot_loss import (
    batched_ottc_loss_bucketized,
    jensen_shannon_divergence as original_upstream_jsd,
)


@dataclass
class SparseTransport:
    batch: torch.Tensor
    frame: torch.Tensor
    occurrence: torch.Tensor
    weight: torch.Tensor


def monotone_transport(mass, valid_mask, target_lengths):
    """Original CDF transport on CUDA; same CDF construction for CPU checks.

    Return only positive-mass pairs, at most B*(T+U). The CUDA call directly
    reuses the archived implementation with dummy features to obtain its
    geometry. Its dummy scalar cost is discarded; actual costs are computed
    with history-conditioned Joint logits and the SAME archived JSD function.
    """
    width = int(target_lengths.max())
    target_mask = torch.arange(width, device=mass.device)[None] < target_lengths[:, None]
    target_mass = target_mask.to(mass.dtype)/target_lengths[:, None]
    if mass.is_cuda:
        _, frame, occurrence, weights = batched_ottc_loss_bucketized(
            mass.new_zeros(*mass.shape, 1), mass.new_zeros(len(mass), width, 1),
            mass, target_mass, amask=valid_mask, bmask=target_mask, euclidian=True)
    else:
        # Device-independent reference path; never rewrites the divergence.
        ca, cb = mass.cumsum(-1), target_mass.cumsum(-1)
        edges = torch.cat([ca, cb], -1).sort(-1).values
        weights = torch.diff(F.pad(edges, (1, 0)), dim=-1)
        frame = torch.searchsorted(ca.contiguous(), edges.contiguous(), right=False)
        occurrence = torch.searchsorted(cb.contiguous(), edges.contiguous(), right=False)
        frame = torch.minimum(frame, valid_mask.sum(-1)[:, None]-1)
        occurrence = torch.minimum(occurrence, target_lengths[:, None]-1)
    batch, slot = torch.where(weights > 0)
    return SparseTransport(batch, frame[batch, slot].long(), occurrence[batch, slot].long(), weights[batch, slot])


def mass_boundaries(mass, count):
    """Internal k/count quantiles as continuous frame-cell coordinates [0,T]."""
    if count <= 1:
        return mass.new_empty(0)
    edges = torch.arange(1, count, device=mass.device, dtype=mass.dtype)/count
    cumulative = mass.cumsum(-1)
    index = torch.searchsorted(cumulative.contiguous(), edges.contiguous()).clamp(max=len(mass)-1)
    before = F.pad(cumulative[:-1], (1,0))[index]
    fraction = (edges-before)/mass[index].clamp_min(torch.finfo(mass.dtype).tiny)
    return index.to(mass.dtype) + fraction.clamp(0,1)


def upstream_cost(logits, labels):
    """Call the frozen upstream reverse-KL-to-mixture cost, not standard JSD."""
    one_hot = F.one_hot(labels, logits.shape[-1]).to(logits.dtype)
    return original_upstream_jsd(logits.softmax(-1), one_hot).sum(-1)


def all_phone_costs(logits):
    """Exact same archived cost for every candidate phone, for greedy decoding."""
    n, classes = logits.shape
    probabilities = logits.softmax(-1)[:, None].expand(n, classes, classes)
    targets = torch.eye(classes, device=logits.device, dtype=logits.dtype)[None].expand(n, classes, classes)
    return original_upstream_jsd(probabilities.reshape(-1, classes), targets.reshape(-1, classes)).sum(-1).reshape(n, classes)


class OccurrencePredictor(nn.Module):
    def __init__(self, classes, embedding_dim, hidden_dim, layers, joint_dim, dropout):
        super().__init__()
        self.bos_id = classes  # Input-only embedding; never a classifier output.
        self.embedding = nn.Embedding(classes+1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.)
        self.projection = nn.Linear(hidden_dim, joint_dim)

    def forward(self, tokens, state=None):
        hidden, state = self.lstm(self.embedding(tokens), state)
        return self.projection(hidden), state


class OTTCTransducerHead(nn.Module):
    def __init__(self, input_dim, phone_ids, *, external_vocab_size=42, embedding_dim=256,
                 predictor_dim=256, predictor_layers=2, joint_dim=256, dropout=.1,
                 length_weight=1., initial_phone_rate=.2, max_output_tokens=512):
        super().__init__()
        if not phone_ids or len(set(phone_ids)) != len(phone_ids) or min(phone_ids) < 0 or max(phone_ids) >= external_vocab_size:
            raise ValueError('phone_ids must be distinct external vocabulary IDs')
        if min(input_dim, embedding_dim, predictor_dim, predictor_layers, joint_dim, max_output_tokens) < 1:
            raise ValueError('dimensions and output cap must be positive')
        if not math.isfinite(length_weight) or length_weight <= 0:
            raise ValueError('positive length weight is required for reference-free decoding')
        if not math.isfinite(initial_phone_rate) or initial_phone_rate <= 0 or not 0 <= dropout < 1:
            raise ValueError('invalid phone rate or dropout')
        self.classes = len(phone_ids)
        self.length_weight, self.max_output_tokens = length_weight, max_output_tokens
        self.register_buffer('phone_ids', torch.tensor(phone_ids, dtype=torch.long))
        mapping = torch.full((external_vocab_size,), -1, dtype=torch.long)
        mapping[self.phone_ids] = torch.arange(self.classes)
        self.register_buffer('external_to_internal', mapping)
        self.mass_head = nn.Linear(input_dim, 1)
        # Mean acoustic state predicts log phone/frame rate. log(T) retains the
        # duration information that mean pooling alone would discard.
        self.count_head = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.count_head.weight)
        nn.init.constant_(self.count_head.bias, math.log(initial_phone_rate))
        self.acoustic_projection = nn.Linear(input_dim, joint_dim)
        self.predictor = OccurrencePredictor(self.classes, embedding_dim, predictor_dim,
            predictor_layers, joint_dim, dropout)
        self.joint = Transducer_joint(joint='sum', nonlinearity=nn.Tanh)
        self.output_projection = nn.Linear(joint_dim, self.classes)

    def acoustic(self, features, valid_mask):
        if features.ndim != 3 or valid_mask.shape != features.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError('expected features [B,T,H] and boolean valid_mask [B,T]')
        if not len(features) or (valid_mask.sum(-1) == 0).any():
            raise ValueError('nonempty batch and positive frame lengths required')
        if ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
            raise ValueError('valid frames must be right-padded prefixes')
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        with torch.autocast(features.device.type, enabled=False):
            value = features.to(dtype).masked_fill(~valid_mask[...,None], 0)
            frame_lengths = valid_mask.sum(-1)
            mass = self.mass_head(value).squeeze(-1).masked_fill(~valid_mask, -torch.inf).softmax(-1)
            pooled = value.sum(1)/frame_lengths[:,None]
            log_count = frame_lengths.to(dtype).log() + self.count_head(pooled).squeeze(-1)
            if not torch.isfinite(mass).all() or not torch.isfinite(log_count).all():
                raise ValueError('nonfinite mass or log count')
        return self.acoustic_projection(features), mass, log_count, frame_lengths

    def joint_logits(self, acoustic, history):
        # Use 4-D inputs following SpeechBrain's public Joint interface. Only
        # selected pairs are evaluated, never the dense T*U*joint_dim lattice.
        hidden = self.joint(acoustic[:,None,None], history[:,None,None])[:,0,0]
        return self.output_projection(hidden)

    def targets(self, labels):
        if labels.ndim != 2 or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError('labels must be an integer [B,U] tensor')
        valid = labels != -100
        if ((~valid[:,:-1]) & valid[:,1:]).any() or (valid.sum(-1) == 0).any():
            raise ValueError('targets must be nonempty, right padded with -100')
        if ((labels[valid] < 0) | (labels[valid] >= len(self.external_to_internal))).any():
            raise ValueError('target outside external vocabulary')
        if (self.external_to_internal[labels[valid]] < 0).any():
            raise ValueError('target is not a phone: blank/pad/unk/EOS are excluded')
        lengths = valid.sum(-1)
        safe = labels[:,:int(lengths.max())].masked_fill(~valid[:,:int(lengths.max())], int(self.phone_ids[0]))
        return self.external_to_internal[safe], lengths

    def forward(self, features, valid_mask, labels):
        acoustic, mass, log_count, _ = self.acoustic(features, valid_mask)
        tokens, lengths = self.targets(labels)
        if len(tokens) != len(features):
            raise ValueError('target batch differs from feature batch')
        previous = F.pad(tokens[:,:-1], (1,0), value=self.predictor.bos_id)
        history, _ = self.predictor(previous)
        with torch.autocast(features.device.type, enabled=False):
            transport = monotone_transport(mass, valid_mask, lengths)
        logits = self.joint_logits(acoustic[transport.batch, transport.frame], history[transport.batch, transport.occurrence])
        with torch.autocast(features.device.type, enabled=False):
            logits = logits.to(mass.dtype)
            costs = upstream_cost(logits, tokens[transport.batch, transport.occurrence])
            ottc_sum = (transport.weight*costs).sum()
            length_sum = F.smooth_l1_loss(log_count, lengths.to(log_count.dtype).log(), reduction='sum')
            loss = (ottc_sum+self.length_weight*length_sum)/len(features)
        return dict(loss=loss, ottc_sum=ottc_sum, length_sum=length_sum, utterance_count=len(features),
            token_count=lengths.sum(), log_count=log_count, mass=mass, transport=transport, pair_logits=logits)

    @torch.no_grad()
    def recognize(self, features, valid_mask):
        if self.training:
            raise ValueError('call eval() before text-free recognition')
        acoustic, mass, log_count, frame_lengths = self.acoustic(features, valid_mask)
        # Round-half-up and an explicit fixed safety cap. No reference count,
        # supplied phones, blank transition, EOS, or repeat-collapse operation.
        bounded = log_count.clamp(min=-20, max=math.log(self.max_output_tokens)).exp()
        counts = torch.floor(bounded+.5).long().clamp(min=1, max=self.max_output_tokens)
        clipped = (log_count > math.log(self.max_output_tokens)) | (log_count < math.log(.5))
        transport = monotone_transport(mass, valid_mask, counts)
        sequences, positions = [], []
        for b, count in enumerate(counts.tolist()):
            previous = torch.full((1,1), self.predictor.bos_id, dtype=torch.long, device=features.device)
            history, state = self.predictor(previous)
            sequence = []
            for u in range(count):
                select = (transport.batch==b) & (transport.occurrence==u)
                if not select.any():
                    raise RuntimeError('positive target bin has no transport support')
                frames = transport.frame[select]
                logits = self.joint_logits(acoustic[b,frames], history[0,0].expand(len(frames),-1))
                with torch.autocast(features.device.type, enabled=False):
                    candidate_cost = all_phone_costs(logits.to(mass.dtype))
                    score = (transport.weight[select,None]*candidate_cost).sum(0)
                token = int(score.argmin())
                sequence.append(int(self.phone_ids[token]))
                previous.fill_(token)
                history, state = self.predictor(previous, state)
            sequences.append(sequence)
            positions.append(mass_boundaries(mass[b,:int(frame_lengths[b])], count))
        return dict(tokens=sequences, boundary_positions=positions, predicted_lengths=counts,
            predicted_log_count=log_count, length_clipped=clipped, mass=mass, frame_lengths=frame_lengths)
