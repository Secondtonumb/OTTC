"""Four phone-set heads over one Wav2Vec2 encoder.

The OT objective uses the original monotone one-dimensional transport and
cost definitions. Reduction is mean over utterances, then mean over heads.
Each head has its own frame-mass predictor and its own target sequence.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model

from ottc.ops_loss.ot_loss import (
    jensen_shannon_divergence,
    jensen_shannon_divergence_standard,
)

HEADS = ('ottc_arpabet', 'ifmdd_arpabet', 'mfa_ipa', 'zipa_ipa')


def transport_loss(logits, targets, frame_weights, frame_lengths, cost='ce'):
    """Sparse monotone OT; padded bins have zero mass and never incur cost."""
    mask = targets >= 0
    target_lengths = mask.sum(-1)
    if (target_lengths == 0).any():
        raise ValueError('Every supervised head must have a nonempty target')
    target_weights = mask.float() / target_lengths[:, None]
    ca, cb = frame_weights.cumsum(-1), target_weights.cumsum(-1)
    knots = torch.cat((ca, cb), -1).sort(-1).values
    ix = torch.searchsorted(ca.contiguous(), knots.contiguous()).clamp_min(0)
    iy = torch.searchsorted(cb.contiguous(), knots.contiguous()).clamp_min(0)
    ix = torch.minimum(ix, frame_lengths[:, None]-1)
    iy = torch.minimum(iy, target_lengths[:, None]-1)
    masses = knots.diff(prepend=torch.zeros_like(knots[:, :1]), dim=-1)
    active = masses > 0
    batch_index = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(ix)
    selected = logits[batch_index[active], ix[active]].float()
    labels = targets[batch_index[active], iy[active]]
    if cost == 'ce':
        values = F.cross_entropy(selected, labels, reduction='none')
    else:
        onehot = F.one_hot(labels, logits.shape[-1]).float()
        if cost == 'jsd_upstream':
            values = jensen_shannon_divergence(selected.softmax(-1), onehot).sum(-1)
        elif cost == 'jsd_standard':
            values = jensen_shannon_divergence_standard(selected, onehot).sum(-1)
        else:
            raise ValueError(cost)
    return (masses[active] * values).sum() / logits.shape[0]


class MultiPhoneConfig(PretrainedConfig):
    model_type = 'wav2vec2-multiphoneset'

    def __init__(self, encoder_config=None, vocabs=None, objective='ctc',
                 head_weights=None, dropout=0.1, freeze_alignment=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or {}
        self.vocabs = vocabs or {}
        self.objective = objective
        self.head_weights = head_weights or {h: 1.0 for h in HEADS}
        self.dropout = dropout
        self.freeze_alignment = freeze_alignment


class MultiPhoneModel(PreTrainedModel):
    config_class = MultiPhoneConfig
    supports_gradient_checkpointing = True
    main_input_name = 'input_values'

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Wav2Vec2Model(Wav2Vec2Config(**config.encoder_config))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.phone_heads = nn.ModuleDict({h: nn.Linear(hidden, len(config.vocabs[h])) for h in HEADS})
        self.mass_heads = nn.ModuleDict({h: nn.Linear(hidden, 1) for h in HEADS}) if config.objective != 'ctc' else nn.ModuleDict()
        self.alignment_encoder = copy.deepcopy(self.encoder) if config.freeze_alignment else None
        self.encoder.feature_extractor._freeze_parameters()
        if self.alignment_encoder is not None:
            self.alignment_encoder.requires_grad_(False)
        self.post_init()

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        for module in self.encoder.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = enable
                if gradient_checkpointing_func is not None:
                    module._gradient_checkpointing_func = gradient_checkpointing_func

    def freeze_alignment(self):
        if self.config.objective == 'ctc':
            return
        if self.alignment_encoder is None:
            self.alignment_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.mass_heads.requires_grad_(False)
        self.config.freeze_alignment = True

    def forward(self, input_values, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
        lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()
        encoded = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(encoded)
        logits = {h: self.phone_heads[h](hidden) for h in HEADS}
        result = {'logits': logits, 'lengths': lengths}
        if labels is None:
            return result
        losses = {}
        mass_hidden = hidden
        if self.alignment_encoder is not None:
            self.alignment_encoder.eval()
            with torch.no_grad():
                mass_hidden = self.alignment_encoder(input_values, attention_mask=attention_mask).last_hidden_state
        frame_mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        for h in HEADS:
            supervised=(labels[h]>=0).any(-1)
            if not supervised.any():
                losses[h]=logits[h].sum()*0
                continue
            target = labels[h][supervised]
            head_logits=logits[h][supervised]
            head_lengths=lengths[supervised]
            if self.config.objective == 'ctc':
                target_lengths = (target >= 0).sum(-1)
                repeats = ((target[:, 1:] == target[:, :-1]) & (target[:, 1:] >= 0)).sum(-1)
                if (target_lengths + repeats > head_lengths).any():
                    raise ValueError(f'CTC target has insufficient frames: {h}')
                with torch.backends.cudnn.flags(enabled=False):
                    losses[h] = F.ctc_loss(head_logits.float().log_softmax(-1).transpose(0, 1),
                                           target[target >= 0], head_lengths, target_lengths,
                                           blank=0, reduction='mean', zero_infinity=False)
            else:
                mass_scores = self.mass_heads[h](mass_hidden[supervised]).squeeze(-1).float()
                weights = mass_scores.masked_fill(~frame_mask[supervised], -torch.inf).softmax(-1)
                losses[h] = transport_loss(head_logits, target, weights, head_lengths, self.config.objective)
        total_weight = sum(self.config.head_weights.values())
        result.update(loss=sum(losses[h]*self.config.head_weights[h] for h in HEADS)/total_weight,
                      head_losses=losses)
        return result
