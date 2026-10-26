"""Repair ModelOutput mapping for Accelerate, preserving the original recipe."""
from transformers.modeling_outputs import CausalLMOutput
from ottc.models.sequence.wav2vec2_ottc_gw import GWTransportMixin as OriginalGWTransportMixin


class GWTransportMixin(OriginalGWTransportMixin):
    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None):
        out = super().forward(input_values, attention_mask=attention_mask,
                              return_dict=return_dict, labels=labels)
        if isinstance(out, CausalLMOutput) and out.loss is not None:
            # Setting an initially absent ModelOutput attribute does not add
            # its mapping entry. Accelerate reconstructs from that mapping.
            # Construct both views together; all tensors/graphs are unchanged.
            return CausalLMOutput(loss=out.loss, logits=out.logits,
                                 hidden_states=out.hidden_states, attentions=out.attentions)
        return out
