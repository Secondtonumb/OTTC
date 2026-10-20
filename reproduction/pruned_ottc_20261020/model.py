"""Opt-in training adapter; use in a new recipe, never patch active models."""

import torch
from torch import nn
from torch.nn import functional as functional

from reproduction.pruned_ottc_20261020.loss import pruned_ottc_loss


class PrunedTrainingOutput(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class _DeferredLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.deferred = False

    def forward(self, hidden):
        if self.deferred:
            return hidden
        return functional.linear(hidden, self.weight, self.bias)


class PrunedOTTCMixin:
    """Combine with wvForOTTC in a new class before constructing the optimizer.

    Training with labels returns a scalar loss and transport metadata, not full
    vocabulary logits. Eval and label-free inference use the unchanged base
    forward. The original head parameter names and checkpoint weights survive.
    Batch means one forward/microbatch, not an accumulated optimizer batch.
    """

    def __init__(self, *args, pruned_chunk_size=None, pruned_jsd_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.lm_head, nn.Linear):
            raise ValueError('pruned OTTC currently requires a linear prediction head')
        if self.forward_type not in ('regular', 'regfrozen'):
            raise ValueError('pruned OTTC supports regular and regfrozen forward types')
        if self.oracle_weights_model_bool:
            raise ValueError('oracle-generated targets require a separate explicit adaptation')
        if self.euclidian or self.jsd not in (False, True):
            raise ValueError('pruned OTTC supports CE and original upstream JSD only')
        chunk_size = (getattr(self.config, 'pruned_chunk_size', 4096)
                      if pruned_chunk_size is None else pruned_chunk_size)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError('pruned_chunk_size must be a positive integer')
        self.config.pruned_chunk_size = chunk_size
        self.config.pruned_vocabulary = 'per-forward target union, blank excluded, full normalization'
        self.config.pruned_cost = 'original_jsd_with_full_background' if self.jsd else 'ce'
        self.lm_head = _DeferredLinear(self.lm_head)
        self.pruned_jsd_function = pruned_jsd_function
        self.last_pruned_vocabulary = None

    def forward(self, input_values=None, attention_mask=None, return_dict=None, labels=None):
        if not self.training or labels is None:
            return super().forward(input_values=input_values, attention_mask=attention_mask,
                                   return_dict=return_dict, labels=labels)
        if self.lm_head.deferred:
            raise RuntimeError('reentrant pruned model forward is unsupported')
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        self.lm_head.deferred = True
        try:
            if self.forward_type == 'regular':
                hidden, frame_mass, frame_mask, _ = self.regular_forward(
                    input_values, attention_mask, return_dict, labels)
            else:
                hidden, frame_mass, frame_mask = self.regular_forward_and_frozen(
                    input_values, attention_mask, return_dict, stop_steps=self.frozen_steps_for_regfrozen)
        finally:
            self.lm_head.deferred = False
        with torch.autocast(hidden.device.type, enabled=False):
            result = pruned_ottc_loss(hidden, self.lm_head.weight, self.lm_head.bias, labels,
                frame_mass, input_mask=frame_mask, cost='jsd' if self.jsd else 'ce',
                chunk_size=self.config.pruned_chunk_size, blank_id=getattr(self.config, 'pad_token_id', 0),
                jsd_function=self.pruned_jsd_function)
        self.last_pruned_vocabulary = result.vocabulary
        self.count += 1
        target_mass = (labels != -100).to(frame_mass.dtype) / result.vocabulary.lengths[:, None]
        if not return_dict:
            return (result.loss,)
        return PrunedTrainingOutput(loss=result.loss, logits=None,
            hidden_states=(frame_mass, frame_mask, target_mass),
            attentions=(result.frame_indices, result.target_indices, result.transport_weights))