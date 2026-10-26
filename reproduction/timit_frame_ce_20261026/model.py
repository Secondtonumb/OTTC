"""Gold-PHN frame CE; audio-only inference, no transport or duration head loss."""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from reproduction.timit_ssl_native10_20261019.common import ScratchOTTC


def frame_loss(logits, targets, mask):
    assert targets.shape == mask.shape == logits.shape[:2]
    assert not (targets.ne(-100) & ~mask.bool()).any()
    valid = targets.ne(-100).sum(-1)
    assert (valid > 0).all()
    values = F.cross_entropy(logits.float().transpose(1, 2), targets,
                             ignore_index=-100, reduction='none')
    # Trainer divides this sum by the actual logical utterance batch size.
    return (values.sum(-1) / valid).sum()


class FrameCE(ScratchOTTC):
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
                raise ValueError('Frame CE requires gold-PHN frame targets during training/loss evaluation')
            loss = frame_loss(logits, frame_targets, mask)
        # Same output-mask position consumed by the frozen evaluator/Trainer.
        # The unused OT duration head is retained frozen for checkpoint identity.
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(None, mask))
