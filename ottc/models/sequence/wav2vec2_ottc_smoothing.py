"""Opt-in smoothing extension; original OTTC sources and outputs stay intact."""
import math

import torch

from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from ottc.ops_loss.temporal_smoothing import temporal_smoothing_loss


class TemporalSmoothingMixin:
    """Add MS-TCN smoothing to a native OTTC model's labeled forward.

    Put this mixin before wvForOTTC (or a device/native-stride subclass) in
    the MRO. Persist weight/tau in wvOTTCConfig, which already accepts extra
    configuration fields. No parameters or state-dict keys are added.
    """
    def __init__(self, config):
        weight = float(getattr(config, 'temporal_smoothing_weight', 0.))
        tau = float(getattr(config, 'temporal_smoothing_tau', 4.))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('temporal_smoothing_weight must be finite and nonnegative')
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(tau * tau):
            raise ValueError('temporal_smoothing_tau must be finite and positive')
        reduction = getattr(config, 'temporal_smoothing_reduction', 'utterance_mean_then_batch_sum')
        if reduction != 'utterance_mean_then_batch_sum':
            raise ValueError('OTTC integration requires utterance_mean_then_batch_sum')
        config.temporal_smoothing_weight = weight
        config.temporal_smoothing_tau = tau
        config.temporal_smoothing_reduction = reduction
        super().__init__(config)
        self._temporal_smoothing_statistics = None

    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None):
        weight = self.config.temporal_smoothing_weight
        if labels is None or weight == 0:
            # Bitwise base behavior when disabled; audio-only inference never
            # computes smoothing, OT transport or reference-conditioned data.
            return super().forward(input_values, attention_mask=attention_mask,
                                   return_dict=return_dict, labels=labels)
        as_dict = return_dict if return_dict is not None else self.config.use_return_dict
        out = super().forward(input_values, attention_mask=attention_mask,
                              return_dict=True, labels=labels)
        ottc = out.loss
        # This mask is already at the encoder's output frame rate, including
        # native-stride models. Neither sample masks nor alpha are used here.
        smooth = temporal_smoothing_loss(out.logits, out.hidden_states[1],
            tau=self.config.temporal_smoothing_tau, reduction='sum')
        weighted = weight * smooth
        out.loss = ottc + weighted
        if self.training and torch.is_grad_enabled():
            # Detached device-side sums; synchronize only when explicitly
            # consumed for logging, not on every forward/microbatch.
            values = torch.stack((ottc.detach(), smooth.detach(), weighted.detach(),
                                  out.loss.detach(), ottc.new_tensor(out.logits.shape[0])))
            previous = self._temporal_smoothing_statistics
            self._temporal_smoothing_statistics = values if previous is None else previous + values
        # Preserve the upstream tuple contract and all structured output keys.
        return out if as_dict else (out.loss, out.logits)

    def pop_temporal_smoothing_statistics(self):
        """Return detached [OTTC, smooth, weighted smooth, total, examples] sums.

        Consumers should reduce these sums across ranks before normalizing.
        Evaluation and audio-only forwards do not enter this accumulator.
        """
        values = self._temporal_smoothing_statistics
        self._temporal_smoothing_statistics = None
        return values


class wvForOTTCTemporalSmoothing(TemporalSmoothingMixin, wvForOTTC):
    """Optional single-head OTTC model; weight defaults to zero."""
