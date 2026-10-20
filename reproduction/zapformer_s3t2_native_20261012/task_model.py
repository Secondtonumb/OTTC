"""Phone task adapter; Zapformer backbone and losses reuse frozen implementations."""
import runtime
import torch
from torch import nn
import train as native
from zapformer_modules import ScaledLinear
from ifmdd_losses import make_ottc_inputs, pad_flat_targets, expand_subphonetic_targets, topology_forward_sum_loss
from archived_ot_loss import batched_ottc_loss_bucketized


def parameters(tiny=False):
    p = native.get_params()
    p.update(vars(native.get_parser().parse_args([])))
    p.update(use_transducer=False, use_ctc=True, use_attention_decoder=False,
             vocab_size=105, blank_id=0, sos_id=1, eos_id=1)
    if tiny:
        p.update(base_dim=8, embed_multiple=4, encoder_multiple='4,4',
                 downsampling_factor='1,2', num_encoder_layers='1,1',
                 feedforward_multiple='2,2', num_heads='2', query_head_dim='8',
                 value_head_dim='8', pos_head_dim='4', conv_params='7,7')
    return p


class FrameProjection(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.projection = ScaledLinear(feature_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, lengths, aux_loss_scale=0.):
        return self.norm(self.projection(x)), lengths


class S3T2(nn.Module):
    topology = 's3_t2'
    frame_shift_seconds = .01

    def __init__(self, objective='jsd', tiny=False):
        super().__init__()
        if objective not in ('jsd', 'framece'):
            raise ValueError(objective)
        self.objective = objective
        p = parameters(tiny)
        self.acoustic = native.get_model(p)
        dim = native.lookup(p, 'embed_dim')
        self.acoustic.encoder_embed = FrameProjection(p.feature_dim, dim)
        self.acoustic.encoder.output_downsampling_factor = 1
        self.acoustic.ctc_output = nn.Sequential(ScaledLinear(dim, 313, initial_scale=.1))
        self.weight_head = ScaledLinear(dim, 1, initial_scale=.1)
        if objective == 'framece':
            self.weight_head.requires_grad_(False)

    def forward(self, features, lengths, tokens, token_lengths, frame_targets=None, aux_loss_scale=0.):
        enc, out_lengths = self.acoustic.forward_encoder(features, lengths, aux_loss_scale=aux_loss_scale)
        assert enc.shape[1] == features.shape[1] and torch.equal(lengths, out_lengths)
        logits = self.acoustic.ctc_output(enc).float()
        logits = logits.index_fill(-1, torch.arange(1, 7, device=logits.device), -1e4)
        with torch.autocast(features.device.type, enabled=False):
            if self.objective == 'framece':
                assert frame_targets is not None
                logp = logits.log_softmax(-1)
                phones = torch.cat([logp[..., :1], torch.logsumexp(logp[..., 1:].reshape(*logp.shape[:-1], 104, 3), -1)], -1)
                loss = nn.functional.nll_loss(phones.reshape(-1, 105), frame_targets.reshape(-1),
                                               ignore_index=-100, reduction='sum')
                return loss, {'frame_ce_nll_sum': loss.detach()}
            expanded, expanded_lengths = expand_subphonetic_targets(tokens, token_lengths, 3)
            padded = pad_flat_targets(expanded, expanded_lengths)
            weights = self.weight_head(enc.float()).squeeze(-1)
            fw, tw, fm, onehot = make_ottc_inputs(logits, out_lengths, padded, expanded_lengths, weights)
            tm = torch.arange(padded.shape[1], device=logits.device)[None] < expanded_lengths[:, None]
            acoustic_ids = torch.cat([torch.zeros(1, device=logits.device, dtype=torch.long), torch.arange(7, 313, device=logits.device)])
            # Exact previously reproduced upstream objective; no replacement JSD formula.
            jsd = batched_ottc_loss_bucketized(logits.index_select(-1, acoustic_ids), onehot.index_select(-1, acoustic_ids),
                                               fw, tw, amask=fm, bmask=tm, jsd=True)[0]
            topology = topology_forward_sum_loss(logits.log_softmax(-1), out_lengths, tokens,
                                                 token_lengths, topology='s3_t2', blank_id=0)
            return jsd + topology, {'jsd_sum': jsd.detach(), 'topology_forward_sum': topology.detach()}
