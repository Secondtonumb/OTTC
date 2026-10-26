"""Monophone CTC -> ContextFiLM(acoustics) -> monophone OTTC.

An opt-in model: existing OTTC classes, checkpoints and recipes are unchanged.
The modulation matches IF-MDD's ContextFiLM, without its subphone topology.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput

from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
from ottc.ops_loss.ot_loss import batched_ottc_loss_bucketized


class ContextFiLM(nn.Module):
    """Content posterior conditions a frame-wise, initially identity transform.

    Adapted from IF-MDD/models/phn_mono_ssl_model_v3_refactored.py:ContextFiLM.
    gamma is bounded; beta is linear, exactly as in that implementation.
    """
    def __init__(self, feature_size, condition_size, hidden_size=None,
                 gamma_scale=0.5, beta_scale=0.5, dropout=0.0):
        super().__init__()
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        hidden_size = hidden_size or feature_size
        self.net = nn.Sequential(
            nn.Linear(condition_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, feature_size * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, condition):
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        gamma = gamma.tanh() * self.gamma_scale
        return features * (1.0 + gamma) + beta * self.beta_scale


class wvOTTCFiLMConfig(wvOTTCConfig):
    model_type = 'wav2vec2-ottc-monophone-film'

    def __init__(self, ctc_loss_weight=1.0, ottc_loss_weight=1.0,
                 film_hidden_size=None, film_gamma_scale=0.5,
                 film_beta_scale=0.5, film_dropout=0.0,
                 film_detach_condition=True, blank_id=0, **kwargs):
        kwargs.setdefault('bins_metric', 'jsd')
        kwargs.setdefault('forward_type', 'regular')
        super().__init__(**kwargs)
        if self.forward_type != 'regular' or self.oracle_weights_model_name is not None:
            raise ValueError('Monophone FiLM requires regular forward and no oracle model')
        for name, value in [('ctc_loss_weight', ctc_loss_weight),
                            ('ottc_loss_weight', ottc_loss_weight),
                            ('film_gamma_scale', film_gamma_scale),
                            ('film_beta_scale', film_beta_scale)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, float(value))
        if ctc_loss_weight <= 0 or ottc_loss_weight <= 0:
            raise ValueError('Both heads require positive loss weights')
        if not isinstance(blank_id, int) or not 0 <= blank_id < self.vocab_size:
            raise ValueError('blank_id must index the shared monophone vocabulary')
        if film_hidden_size is not None and (not isinstance(film_hidden_size, int) or film_hidden_size <= 0):
            raise ValueError('film_hidden_size must be a positive integer')
        if not math.isfinite(film_dropout) or not 0 <= film_dropout < 1:
            raise ValueError('film_dropout must be in [0, 1)')
        self.film_hidden_size = film_hidden_size
        self.film_gamma_scale = float(film_gamma_scale)
        self.film_beta_scale = float(film_beta_scale)
        self.film_dropout = float(film_dropout)
        self.film_detach_condition = bool(film_detach_condition)
        self.blank_id = blank_id
        self.phone_inventory = 'monophone'
        self.loss_reduction = 'weighted sum of native OTTC batch sum and CTC batch sum'


class wvForOTTCFiLM(wvForOTTC):
    """Two classifiers sharing one monophone vocabulary, including CTC blank.

    CTC reads original acoustics; OTTC classifier and learned frame mass read
    FiLM-modulated acoustics. Both use the same unexpanded phone targets. No
    transcript enters the FiLM condition, including during training.
    """
    config_class = wvOTTCFiLMConfig

    def __init__(self, config):
        super().__init__(config)
        size = self.wav2vec2.config.hidden_size
        self.ctc_head = nn.Linear(size, config.vocab_size)
        self.context_film = ContextFiLM(size, config.vocab_size,
            config.film_hidden_size, config.film_gamma_scale,
            config.film_beta_scale, config.film_dropout)
        self.float()
        self._film_loss_statistics = None

    def _encode_heads(self, input_values, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
        h = self.dropout(self.wav2vec2(input_values,
            attention_mask=attention_mask, return_dict=True).last_hidden_state)
        lengths = self._get_feat_extract_output_lengths(
            self.wave2vec2_config, attention_mask.sum(-1)).long()
        if (lengths <= 0).any() or (lengths > h.shape[1]).any():
            raise ValueError('Invalid encoder output lengths')
        valid = torch.arange(h.shape[1], device=h.device)[None, :] < lengths[:, None]
        ctc_logits = self.ctc_head(h)
        # Keep the posterior and native OTTC probability arithmetic in FP32,
        # even when the encoder/linear layers use approved autocast compute.
        with torch.autocast(device_type=h.device.type, enabled=False):
            condition = ctc_logits.float().softmax(-1)
        if self.config.film_detach_condition:
            condition = condition.detach()
        modulated = self.context_film(h, condition)
        logits = self.lm_head(modulated)
        mass_logits = self.lm_weight(modulated).squeeze(-1).float()
        mass = mass_logits.masked_fill(~valid, -torch.inf).softmax(-1)
        return logits, ctc_logits, mass, valid

    def ctc_forward(self, input_values, attention_mask=None):
        """Audio-only auxiliary CTC logits, for independent CTC decoding."""
        _, logits, _, valid = self._encode_heads(input_values, attention_mask)
        return CausalLMOutput(logits=logits, hidden_states=(valid,))

    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None):
        as_dict = self.config.use_return_dict if return_dict is None else return_dict
        logits, ctc_logits, mass, valid = self._encode_heads(input_values, attention_mask)
        loss, attentions, target_mass = None, None, None
        if labels is not None:
            if labels.ndim != 2 or labels.shape[0] != logits.shape[0]:
                raise ValueError('labels must have shape [batch, target phones]')
            target_valid = labels != -100
            targets = labels[target_valid]
            lengths = target_valid.sum(-1)
            if (lengths == 0).any() or (targets < 0).any() or (targets >= self.config.vocab_size).any():
                raise ValueError('Each target must contain valid monophone IDs; padding is -100')
            if (targets == self.config.blank_id).any():
                raise ValueError('Monophone targets must not contain CTC blank')
            # Both native OT and CTC require right-padded, ordered targets.
            if (target_valid[:, 1:] & ~target_valid[:, :-1]).any():
                raise ValueError('labels must be right padded with -100')
            repeats = ((labels[:, 1:] == labels[:, :-1]) & target_valid[:, 1:]).sum(-1)
            if (lengths + repeats > valid.sum(-1)).any():
                raise ValueError('Insufficient encoder frames for CTC targets, including repeats')
            safe_labels = labels.masked_fill(~target_valid, 0)
            with torch.autocast(device_type=logits.device.type, enabled=False):
                one_hot = F.one_hot(safe_labels, self.config.vocab_size).float()
                target_mass = target_valid.float() / lengths[:, None]
                ottc, ix, iy, transport = batched_ottc_loss_bucketized(
                    logits.float(), one_hot, mass, target_mass,
                    amask=valid, bmask=target_valid, euclidian=self.euclidian,
                    jsd=self.jsd, divergence=self.divergence)
                ctc = F.ctc_loss(ctc_logits.float().log_softmax(-1).transpose(0, 1),
                    targets, valid.sum(-1), lengths, blank=self.config.blank_id,
                    reduction='sum', zero_infinity=False)
                loss = self.config.ottc_loss_weight * ottc + self.config.ctc_loss_weight * ctc
            attentions = (ix, iy, transport)
            if self.training and torch.is_grad_enabled():
                values = torch.stack((ottc.detach(), ctc.detach(), loss.detach(),
                                      loss.new_tensor(labels.shape[0])))
                old = self._film_loss_statistics
                self._film_loss_statistics = values if old is None else old + values
        if not as_dict:
            return (logits,) if loss is None else (loss, logits)
        # Preserve existing OTTC scoring/Trainer output contracts. CTC logits
        # are available explicitly via ctc_forward, not an extra prediction.
        hidden = (mass, valid) if target_mass is None else (mass, valid, target_mass)
        return CausalLMOutput(loss=loss, logits=logits,
                              hidden_states=hidden, attentions=attentions)

    def pop_film_loss_statistics(self):
        """Detached [raw OTTC sum, raw CTC sum, total sum, utterance count].

        Training loggers should reduce across ranks before normalizing. This
        is separate from evaluation logging, which remains local-only.
        """
        values = self._film_loss_statistics
        self._film_loss_statistics = None
        return values
