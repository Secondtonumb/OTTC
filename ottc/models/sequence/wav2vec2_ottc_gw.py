"""Opt-in experimental GW objectives. No change to historical OTTC classes."""
import torch
from transformers.modeling_outputs import CausalLMOutput

from ottc.ops_loss.gw_transport import (
    GWOptions, ag_cost, asot_plan, objective_terms, plan_diagnostics, structure,
)


class GWTransportMixin:
    def __init__(self, config):
        self.gw_options = GWOptions(**config.gw_options)
        if config.forward_type != 'regular' or config.bins_metric != 'ag':
            raise ValueError('Initial GW experiment supports regular mono AG only')
        super().__init__(config)
        self._gw_statistics = None

    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None):
        opts = self.gw_options
        as_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is None or (opts.mode == 'monotone' and opts.weight == 0):
            return super().forward(input_values, attention_mask=attention_mask,
                                   return_dict=return_dict, labels=labels)
        if self.oracle_weights_model_bool:
            raise ValueError('No oracle targets in GW experiment')
        # Mode 2 reuses the EXACT base loss and its sparse coupling weights.
        # Mode 3 calls the same audio encoder, but replaces the training plan.
        out = super().forward(input_values, attention_mask=attention_mask,
                              return_dict=True, labels=labels if opts.mode == 'monotone' else None)
        masses, mask = out.hidden_states[:2]
        totals, statistics = [], []
        with torch.autocast(device_type=out.logits.device.type, enabled=False):
            for i in range(len(labels)):
                valid = mask[i].bool()
                target = labels[i][labels[i] >= 0]
                if not len(target) or int(valid.sum()) < 1:
                    raise ValueError('Empty frame or target sequence')
                a = masses[i][valid].float()
                b = torch.ones(len(target), device=a.device, dtype=a.dtype) / len(target)
                if opts.mode == 'monotone':
                    ix, iy, weights = (value[i] for value in out.attentions)
                    # Include padded frame slots for scatter, then drop them.
                    flat = weights.float().new_zeros(mask.shape[1] * len(target))
                    q = flat.scatter_add(0, ix.long() * len(target) + iy.long(), weights.float())
                    q = q.reshape(mask.shape[1], len(target))[valid]
                    gw = structure(q, opts.window_frames)
                    totals.append(opts.weight * gw)
                    terms = dict(gw=gw)
                else:
                    cost = ag_cost(out.logits[i][valid].float(), target)
                    q = asot_plan(cost, a, b, opts)
                    terms = objective_terms(q, cost, b, opts)
                    totals.append(terms['total'])
                if self.training and torch.is_grad_enabled():
                    stats = {key: value.detach() for key, value in terms.items()}
                    stats.update(plan_diagnostics(q, a, b))
                    statistics.append(stats)
        addition = torch.stack(totals).sum()
        if opts.mode == 'monotone':
            base_loss = out.loss
            out.loss = base_loss + addition
        else:
            out.loss = addition
            # Dense ASOT plans are not the historical sparse attentions tuple.
            # Avoid passing a reference-conditioned plan as inference output.
            out.attentions = None
        if statistics:
            sums = {key: sum(torch.as_tensor(s[key], device=out.loss.device) for s in statistics)
                    for key in statistics[0]}
            sums.update(examples=out.loss.new_tensor(len(labels)), loss=out.loss.detach())
            if opts.mode == 'monotone':
                sums['ag'] = base_loss.detach()
            previous = self._gw_statistics
            self._gw_statistics = sums if previous is None else {key: previous[key] + value for key, value in sums.items()}
        return out if as_dict else (out.loss, out.logits)

    def pop_gw_statistics(self):
        values, self._gw_statistics = self._gw_statistics, None
        return values
