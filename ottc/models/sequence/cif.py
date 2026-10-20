"""Continuous integrate-and-fire and a phone decoder, independent of the encoder.

Original implementation from the interval-overlap definition of CIF (1905.11235).
References: MingLunHan/CIF-PyTorch and CIF-HieraDist. No upstream code copied.
Mass, integration and loss arithmetic stay in FP32 (FP64 for gradient checks).
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CIFOutput:
    embeddings: Tensor
    lengths: Tensor
    valid_mask: Tensor
    quantity: Tensor
    weights: Tensor
    # End of each emitted segment, in continuous frame-cell coordinates [0,T].
    fire_positions: Tensor


def integrate_and_fire(
    features: Tensor,
    weights: Tensor,
    valid_mask: Tensor,
    target_lengths: Tensor | None = None,
    *,
    tail_threshold: float = 0.5,
    chunk_size: int = 64,
) -> CIFOutput:
    """Threshold=1; explicit target_lengths requests supervised length scaling.

    Without target_lengths, use predicted counts and retain a normalized tail
    only when its mass is strictly greater than tail_threshold. Supports several
    firings in one frame after scaling; no firing is inferred from feature values.
    Input masks are True for valid frames and must be right-padded prefixes.
    """
    if features.ndim != 3 or weights.shape != features.shape[:2]:
        raise ValueError("expected features [B,T,D] and weights [B,T]")
    if not features.shape[0] or not features.shape[1]:
        raise ValueError("batch and frame dimensions must be nonempty")
    if valid_mask.shape != weights.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B,T], True for valid frames")
    if not 0 <= tail_threshold < 1 or chunk_size < 1:
        raise ValueError("invalid tail threshold or chunk size")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
        raise ValueError("valid frames must form a prefix")
    frame_lengths = valid_mask.sum(-1)
    if (frame_lengths == 0).any():
        raise ValueError("each utterance needs at least one valid frame")
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    with torch.autocast(features.device.type, enabled=False):
        h = features.to(dtype).masked_fill(~valid_mask[..., None], 0)
        alpha = weights.to(dtype).masked_fill(~valid_mask, 0)
        quantity = alpha.sum(-1)
        if target_lengths is not None:
            if target_lengths.shape != quantity.shape or target_lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("target_lengths must be integer [B]")
            if (target_lengths < 0).any() or ((quantity == 0) & (target_lengths > 0)).any():
                raise ValueError("positive targets require positive input mass")
            alpha = alpha * (target_lengths / quantity.clamp_min(torch.finfo(dtype).tiny))[:, None]
            lengths = target_lengths.to(device=features.device, dtype=torch.long)
        else:
            full = quantity.floor().long()
            lengths = full + ((quantity - full) > tail_threshold).long()
        end = alpha.cumsum(-1)
        start = end - alpha
        width = int(lengths.max().item())
        pieces = []
        positions = []
        # Chunk tokens so the overlap tensor is B x chunk_size x T, never BxSxT.
        for offset in range(0, width, chunk_size):
            bins = torch.arange(offset, min(offset + chunk_size, width), device=h.device, dtype=dtype)
            overlap = (
                torch.minimum(end[:, None], bins[None, :, None] + 1)
                - torch.maximum(start[:, None], bins[None, :, None])
            ).clamp_min(0)
            active = bins[None] < lengths[:, None]
            # Complete bins have unit mass. The inference tail is normalized.
            denominator = torch.ones_like(bins[None] + quantity[:, None])
            if target_lengths is None:
                denominator = (quantity[:, None] - bins[None]).clamp(max=1).clamp_min(torch.finfo(dtype).tiny)
            values = torch.bmm(overlap, h) / denominator[..., None]
            pieces.append(values.masked_fill(~active[..., None], 0))
            crossing = (bins[None] + 1).expand(len(h), -1).contiguous()
            index = torch.searchsorted(end.contiguous(), crossing).clamp(max=h.shape[1] - 1)
            before = start.gather(1, index)
            mass = alpha.gather(1, index)
            pos = index.to(dtype) + ((crossing - before) / mass.clamp_min(torch.finfo(dtype).tiny)).clamp(0, 1)
            # The scaled final bin may differ from one by floating point roundoff.
            at_tail = crossing > end[:, -1:]
            pos = torch.where(at_tail, frame_lengths[:, None].to(dtype), pos)
            positions.append(pos.masked_fill(~active, 0))
        valid = torch.arange(width, device=h.device)[None] < lengths[:, None]
        if pieces:
            embedding = torch.cat(pieces, 1)
            fire_positions = torch.cat(positions, 1)
        else:
            embedding = h[:, :0] + alpha[:, :0, None]  # preserve a zero gradient path
            fire_positions = alpha[:, :0]
        return CIFOutput(embedding, lengths, valid, quantity, alpha, fire_positions)


class CIFMiddleware(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.weight_proj = nn.Linear(hidden_dim, 1)

    def forward(self, features, valid_mask, target_lengths=None):
        # Mask before convolution so padded hidden states cannot change valid alpha.
        hidden = features.masked_fill(~valid_mask[..., None], 0)
        hidden = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(F.relu(self.norm(hidden)))
        alpha = self.weight_proj(hidden).squeeze(-1).float().sigmoid().clamp_min(1e-6)
        return integrate_and_fire(features, alpha, valid_mask, target_lengths)


class CIFPhoneHead(nn.Module):
    """Paper-style AR decoder: (previous token, previous CIF) -> SAN -> current CIF.

    Uses one private BOS embedding; output phone inventory is unchanged. Targets
    contain phones only, with -100 padding, no EOS. CTC is an optional independent
    frame classifier. Coefficient zero builds no CTC classifier at all.
    """
    def __init__(self, input_dim, vocab_size, *, decoder_dim=256, decoder_layers=2,
                 decoder_heads=4, decoder_ffn=2048, dropout=0.1,
                 quantity_weight=1.0, ctc_weight=0.0, ottc_weight=0.0, label_smoothing=0.1, blank_id=0):
        super().__init__()
        if min(input_dim, vocab_size, decoder_dim, decoder_layers, decoder_heads, decoder_ffn) < 1:
            raise ValueError("dimensions must be positive")
        if decoder_dim % decoder_heads or decoder_dim % 2:
            raise ValueError("decoder_dim must be even and divisible by decoder_heads")
        if not all(math.isfinite(w) and w >= 0 for w in (quantity_weight, ctc_weight, ottc_weight)):
            raise ValueError("loss weights must be finite and nonnegative")
        if ctc_weight and ottc_weight:
            raise ValueError("choose CTC or OTTC auxiliary supervision, not both")
        if not 0 <= label_smoothing < 1 or not 0 <= blank_id < vocab_size:
            raise ValueError("invalid smoothing or blank id")
        self.vocab_size, self.bos_id = vocab_size, vocab_size
        self.blank_id = blank_id
        self.quantity_weight, self.ctc_weight = quantity_weight, ctc_weight
        self.ottc_weight = ottc_weight
        self.label_smoothing = label_smoothing
        self.cif = CIFMiddleware(input_dim, decoder_dim, dropout=dropout)
        self.acoustic_proj = nn.Linear(input_dim, decoder_dim)
        self.token_embedding = nn.Embedding(vocab_size + 1, decoder_dim)
        self.input_proj = nn.Linear(2 * decoder_dim, decoder_dim)
        layer = nn.TransformerEncoderLayer(decoder_dim, decoder_heads, decoder_ffn,
                                           dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, decoder_layers, norm=nn.LayerNorm(decoder_dim),
                                             enable_nested_tensor=False)
        self.output_proj = nn.Linear(2 * decoder_dim, vocab_size)
        self.ctc_head = nn.Linear(input_dim, vocab_size) if ctc_weight else None
        self.ottc_head = nn.Linear(input_dim, vocab_size) if ottc_weight else None
        self.ottc_mass_head = nn.Linear(input_dim, 1) if ottc_weight else None

    def decode(self, cif, previous_tokens):
        width = previous_tokens.shape[1]
        if width == 0:
            return cif.embeddings.new_empty(len(previous_tokens), 0, self.vocab_size)
        current = self.acoustic_proj(cif.embeddings[:, :width].to(self.acoustic_proj.weight.dtype))
        previous = F.pad(current[:, :-1], (0, 0, 1, 0))
        hidden = self.input_proj(torch.cat([self.token_embedding(previous_tokens), previous], -1))
        pos = torch.arange(width, device=hidden.device, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, hidden.shape[-1], 2, device=hidden.device).float()
                          * (-math.log(10000) / hidden.shape[-1]))
        encoding = torch.stack([(pos * scale).sin(), (pos * scale).cos()], -1).flatten(-2)
        hidden = hidden + encoding[None].to(hidden.dtype)
        causal = torch.ones(width, width, device=hidden.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(hidden, mask=causal)
        return self.output_proj(torch.cat([hidden, current], -1))

    def forward(self, features, valid_mask, labels):
        if labels.ndim != 2 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must be [B,S] with -100 padding")
        mask = labels != -100
        if ((labels[mask] < 0) | (labels[mask] >= self.vocab_size)).any():
            raise ValueError("invalid label id")
        if (labels[mask] == self.blank_id).any() or ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("targets must be right-padded phones, without blank")
        lengths = mask.sum(-1)
        if (lengths == 0).any():
            raise ValueError("empty phone targets are not supported")
        labels = labels[:, :int(lengths.max())]
        cif = self.cif(features, valid_mask, lengths)
        previous = F.pad(labels[:, :-1], (1, 0), value=self.bos_id).masked_fill(
            F.pad(labels[:, :-1], (1, 0), value=self.bos_id) == -100, self.blank_id)
        logits = self.decode(cif, previous)
        with torch.autocast(features.device.type, enabled=False):
            ce_sum = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                                     ignore_index=-100, reduction='sum', label_smoothing=self.label_smoothing)
            quantity_sum = (cif.quantity - lengths).abs().sum()
            ctc_sum = ce_sum.new_zeros(())
            ottc_sum = ce_sum.new_zeros(())
            if self.ctc_head is not None:
                required = lengths + ((labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] != -100)).sum(-1)
                if (required > valid_mask.sum(-1)).any():
                    raise ValueError("CTC input too short, including repeated target phones")
                log_probs = self.ctc_head(features.float()).log_softmax(-1).transpose(0, 1)
                with torch.backends.cudnn.flags(enabled=False):
                    ctc_sum = F.ctc_loss(log_probs, labels[labels != -100], valid_mask.sum(-1), lengths,
                                         blank=self.blank_id, reduction='sum', zero_infinity=False)
            if self.ottc_head is not None:
                # Directly reuse the frozen, previously reproduced upstream OTTC.
                # Its jsd=True is reverse KL to the mixture, not standard JSD.
                # This source uses CUDA internally; no divergence is reimplemented.
                from reproduction.zapformer_s3t2_native_20261012.archived_ot_loss import batched_ottc_loss_bucketized
                frame_logits = self.ottc_head(features.float())
                mass_logits = self.ottc_mass_head(features.float()).squeeze(-1)
                mass = mass_logits.masked_fill(~valid_mask, -torch.inf).softmax(-1)
                target_mask = labels != -100
                target_mass = target_mask.float() / lengths[:, None]
                targets = F.one_hot(labels.masked_fill(~target_mask, self.blank_id), self.vocab_size).float()
                ottc_sum = batched_ottc_loss_bucketized(frame_logits, targets, mass, target_mass,
                    amask=valid_mask, bmask=target_mask, jsd=True)[0]
            denominator = lengths.sum()
            loss = (ce_sum + self.quantity_weight * quantity_sum + self.ctc_weight * ctc_sum) / denominator
            # OT already normalizes each utterance's transport mass to one.
            # Keep that native sequence scale; do not divide it by phone count.
            loss = loss + self.ottc_weight * ottc_sum / len(features)
        return dict(loss=loss, logits=logits, ce_sum=ce_sum, quantity_sum=quantity_sum,
                    ctc_sum=ctc_sum, ottc_sum=ottc_sum, token_count=denominator, cif=cif)

    @torch.no_grad()
    def recognize(self, features, valid_mask):
        if self.training:
            raise ValueError("call eval() before text-free recognition")
        cif = self.cif(features, valid_mask)  # no reference labels or lengths
        tokens = torch.full(cif.valid_mask.shape, self.blank_id, dtype=torch.long, device=features.device)
        previous = torch.full((len(features), 1), self.bos_id, device=features.device, dtype=torch.long)
        for index in range(tokens.shape[1]):
            logits = self.decode(cif, previous)[:, -1].float()
            logits[:, self.blank_id] = -torch.inf
            prediction = logits.argmax(-1).masked_fill(~cif.valid_mask[:, index], self.blank_id)
            tokens[:, index] = prediction
            previous = torch.cat([previous, prediction[:, None]], 1)
        return tokens, cif
