"""Monotone-plan GW with existing OTTC CE/AG/standard-JSD/Jeffreys costs.

Only the existing base cost selector changes between sweep conditions.
The historical GW/ASOT classes and active resume contracts are untouched.
"""
import torch
from transformers.modeling_outputs import CausalLMOutput
from ottc.ops_loss.gw_transport import GWOptions, structure, plan_diagnostics


class MonotoneGWTransportMixin:
    def __init__(self, config):
        self.gw_options = GWOptions(**config.gw_options)
        if self.gw_options.mode != 'monotone' or config.forward_type != 'regular':
            raise ValueError('Only regular monotone OTTC is supported')
        if config.bins_metric not in ('crossentropy', 'ag', 'jsd_standard', 'jdiv'):
            raise ValueError('Use an explicitly implemented existing OTTC cost')
        super().__init__(config)
        self._gw_statistics = None

    def forward(self, input_values, attention_mask=None, return_dict=None, labels=None):
        if labels is None or self.gw_options.weight == 0:
            return super().forward(input_values, attention_mask=attention_mask,
                                   return_dict=return_dict, labels=labels)
        if self.oracle_weights_model_bool:
            raise ValueError('No oracle-conditioned target masses in this experiment')
        as_dict = return_dict if return_dict is not None else self.config.use_return_dict
        out = super().forward(input_values, attention_mask=attention_mask, return_dict=True, labels=labels)
        masses, mask = out.hidden_states[:2]
        penalties, statistics = [], []
        with torch.autocast(device_type=out.logits.device.type, enabled=False):
            for i in range(len(labels)):
                valid = mask[i].bool()
                count = int((labels[i] >= 0).sum())
                if count < 1 or int(valid.sum()) < 1:
                    raise ValueError('Empty frame/target sequence')
                a = masses[i][valid].float()
                b = a.new_ones(count) / count
                ix, iy, weights = (value[i] for value in out.attentions)
                q = weights.float().new_zeros(mask.shape[1] * count).scatter_add(
                    0, ix.long() * count + iy.long(), weights.float())
                q = q.reshape(mask.shape[1], count)[valid]
                gw = structure(q, self.gw_options.window_frames)
                penalties.append(gw)
                if self.training and torch.is_grad_enabled():
                    stats = plan_diagnostics(q, a, b)
                    stats['gw'] = gw.detach()
                    statistics.append(stats)
        weighted = self.gw_options.weight * torch.stack(penalties).sum()
        loss = out.loss + weighted
        if statistics:
            sums = {key: sum(torch.as_tensor(s[key], device=loss.device) for s in statistics)
                    for key in statistics[0]}
            sums.update(examples=loss.new_tensor(len(labels)), loss=loss.detach(),
                        base_cost=out.loss.detach(), weighted_gw=weighted.detach())
            previous = self._gw_statistics
            self._gw_statistics = sums if previous is None else {key: previous[key]+value for key,value in sums.items()}
        if not as_dict:
            return loss, out.logits
        return CausalLMOutput(loss=loss, logits=out.logits,
                              hidden_states=out.hidden_states, attentions=out.attentions)

    def pop_gw_statistics(self):
        values, self._gw_statistics = self._gw_statistics, None
        return values
