"""10ms MFA S3T2 objective with archived JSD and recipe auxiliary gradients."""
import runtime
import torch
from torch import nn
from train import get_params, get_parser, get_model, lookup
from zapformer_modules import ScaledLinear
from ifmdd_losses import (make_ottc_inputs, pad_flat_targets,
                         expand_subphonetic_targets, topology_forward_sum_loss)
from archived_ot_loss import batched_ottc_loss_bucketized


def parameters(tiny=False):
    p = get_params()
    p.update(vars(get_parser().parse_args([])))
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


class MFAZapformer(nn.Module):
    frame_shift_seconds = .01

    def __init__(self, tiny=False):
        super().__init__()
        p = parameters(tiny)
        self.acoustic = get_model(p)
        dim = lookup(p, 'embed_dim')
        self.acoustic.encoder_embed = FrameProjection(p.feature_dim, dim)
        self.acoustic.encoder.output_downsampling_factor = 1
        self.acoustic.ctc_output = nn.Sequential(ScaledLinear(dim, 313, initial_scale=.1))
        self.weight_head = ScaledLinear(dim, 1, initial_scale=.1)

    def forward(self, features, lengths, targets, target_lengths, aux_loss_scale=0.):
        enc, out_lengths = self.acoustic.forward_encoder(features, lengths, aux_loss_scale=aux_loss_scale)
        if enc.shape[1] != features.shape[1] or not torch.equal(lengths, out_lengths):
            raise RuntimeError('10ms frame contract violated')
        logits = self.acoustic.ctc_output(enc).float()
        forbidden = torch.arange(1, 7, device=logits.device)
        logits = logits.index_fill(-1, forbidden, -1e4)
        if targets.ndim != 1 or int(target_lengths.sum()) != targets.numel():
            raise ValueError('Expected concatenated MFA phone targets')
        if bool(((targets < 3) | (targets >= 105)).any()) or bool((target_lengths <= 0).any()):
            raise ValueError('Invalid MFA targets')
        with torch.autocast(features.device.type, enabled=False):
            expanded, expanded_lengths = expand_subphonetic_targets(targets, target_lengths, 3)
            padded = pad_flat_targets(expanded, expanded_lengths)
            weights = self.weight_head(enc.float()).squeeze(-1)
            fw, tw, fm, onehot = make_ottc_inputs(logits, out_lengths, padded, expanded_lengths, weights)
            tm = torch.arange(padded.shape[1], device=logits.device)[None] < expanded_lengths[:, None]
            acoustic_ids = torch.cat([torch.zeros(1, device=logits.device, dtype=torch.long),
                                      torch.arange(7, 313, device=logits.device)])
            jsd = batched_ottc_loss_bucketized(
                logits.index_select(-1, acoustic_ids), onehot.index_select(-1, acoustic_ids),
                fw, tw, amask=fm, bmask=tm, jsd=True)[0]
            topology = topology_forward_sum_loss(logits.log_softmax(-1), out_lengths,
                                                 targets, target_lengths, topology='s3_t2', blank_id=0)
        return jsd+topology, logits, out_lengths, jsd.detach(), topology.detach()
