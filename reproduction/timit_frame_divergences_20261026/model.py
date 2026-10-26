"""Gold-PHN frame divergences, reusing the original repository cost functions."""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from reproduction.timit_ssl_native10_20261019.common import ScratchOTTC
from ottc.ops_loss.ot_loss import (
    jensen_shannon_divergence_standard, arithmetic_geometric_divergence, jeffreys_divergence,
)


def per_frame(logits, targets, cost):
    # Explicit FP32 divergence arithmetic under BF16 encoder autocast.
    logits = logits.float()
    q = F.one_hot(targets.clamp_min(0), logits.shape[-1]).float()
    if cost == 'jsd_standard':
        values = jensen_shannon_divergence_standard(logits, q).sum(-1)
    elif cost == 'jdiv':
        values = jeffreys_divergence(logits, q, eps=1e-12).sum(-1)
    elif cost == 'ag':
        values = arithmetic_geometric_divergence(logits.softmax(-1), q).sum(-1)
    else:
        raise ValueError(cost)
    # Never silently change the historical AG formula on softmax underflow.
    if not torch.isfinite(values).all():
        raise FloatingPointError(f'Nonfinite frame {cost}; preserve formula and inspect logits')
    return values


def frame_loss(logits, targets, mask, cost):
    assert targets.shape == mask.shape == logits.shape[:2]
    assert not (targets.ne(-100) & ~mask.bool()).any()
    valid = targets.ne(-100)
    counts = valid.sum(-1)
    assert (counts > 0).all()
    # Only supervised positions enter any logarithm or divergence.
    values = per_frame(logits[valid], targets[valid], cost)
    utterances = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(targets)[valid]
    return (torch.zeros(logits.shape[0], device=logits.device).scatter_add_(0, utterances, values) / counts).sum()


class FrameDivergence(ScratchOTTC):
    def forward(self, input_values, attention_mask=None, return_dict=None,
                labels=None, frame_targets=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
        hidden = self.wav2vec2(input_values, attention_mask=attention_mask,
                               output_hidden_states=True, return_dict=True)[0]
        logits = self.lm_head(self.dropout(hidden))
        lengths = self._get_feat_extract_output_lengths(
            self.wave2vec2_config, attention_mask.sum(-1)).long()
        mask = (torch.arange(logits.shape[1], device=logits.device)[None] < lengths[:, None]).float()
        loss = None
        if labels is not None:
            if frame_targets is None:
                raise ValueError('Frame divergence requires gold-PHN frame targets during training/loss evaluation')
            loss = frame_loss(logits, frame_targets, mask, self.config.frame_divergence)
        # Same output-mask position consumed by the frozen evaluator/Trainer.
        # The unused OT duration head is retained frozen for checkpoint identity.
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(None, mask))
