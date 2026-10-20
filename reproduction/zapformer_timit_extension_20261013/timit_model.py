import bootstrap
import torch
from torch import nn
import torch.nn.functional as F
import train as native
from task_model import parameters, FrameProjection
from zapformer_modules import ScaledLinear
from archived_ot_loss import batched_ottc_loss_bucketized


class TimitOTTC(nn.Module):
    frame_shift_seconds = .01

    def __init__(self, cost='jsd', tiny=False):
        super().__init__()
        if cost not in ('jsd', 'ce'):
            raise ValueError(cost)
        self.cost = cost
        p = parameters(tiny)
        self.acoustic = native.get_model(p)
        dim = native.lookup(p, 'embed_dim')
        self.acoustic.encoder_embed = FrameProjection(p.feature_dim, dim)
        self.acoustic.encoder.output_downsampling_factor = 1
        self.acoustic.ctc_output = nn.Sequential(ScaledLinear(dim, 42, initial_scale=.1))
        self.weight_head = ScaledLinear(dim, 1, initial_scale=.1)

    def scores(self, features, lengths, aux_loss_scale=0.):
        hidden, out_lengths = self.acoustic.forward_encoder(features, lengths, aux_loss_scale=aux_loss_scale)
        if hidden.shape[1] != features.shape[1] or not torch.equal(lengths, out_lengths):
            raise ValueError('Output must preserve the 10ms grid')
        logits = self.acoustic.ctc_output(hidden).float()
        valid = torch.arange(hidden.shape[1], device=hidden.device)[None] < lengths[:, None]
        # Keep normalized OT masses/CDFs in FP32 even under BF16 encoder AMP.
        # This matches the existing native OTTC adapter's mass branch.
        with torch.autocast(features.device.type, enabled=False):
            mass_logits = self.weight_head(hidden.float()).squeeze(-1)
            mass = mass_logits.masked_fill(~valid, -torch.inf).softmax(-1)
        return logits, mass, valid

    def forward(self, features, lengths, tokens, token_lengths, aux_loss_scale=0.):
        logits, mass, valid = self.scores(features, lengths, aux_loss_scale)
        return self.criterion(logits, mass, valid, tokens, token_lengths)

    def criterion(self, logits, mass, valid, tokens, token_lengths):
        padded = torch.zeros(len(token_lengths), int(token_lengths.max()), device=tokens.device, dtype=torch.long)
        cursor = 0
        for i, n in enumerate(token_lengths.tolist()):
            padded[i, :n] = tokens[cursor:cursor+n]
            cursor += n
        target_mask = torch.arange(padded.shape[1], device=tokens.device)[None] < token_lengths[:, None]
        beta = target_mask.float() / token_lengths[:, None]
        # Exact archived upstream CE/JSD routine: no replacement divergence,
        # no substate expansion, no CTC or forward-sum term.
        with torch.autocast(logits.device.type, enabled=False):
            loss = batched_ottc_loss_bucketized(logits, F.one_hot(padded, 42).float(), mass, beta,
                                               amask=valid, bmask=target_mask, jsd=self.cost == 'jsd')[0]
        return loss
