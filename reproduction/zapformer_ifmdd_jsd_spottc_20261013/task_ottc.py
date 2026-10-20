"""Native Zapformer with explicit IF-MDD standard-JSD topology OTTC."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'zapformer_s3t2ss_native_20261012'
sys.path.insert(0, str(NATIVE))
import runtime
import torch
from torch import nn
import train as native
from task_model import parameters, FrameProjection
from ifmdd_losses import pad_flat_targets
from speechbrain.decoders.ctc import ctc_greedy_decode

sys.path.insert(0, str(HERE / 'vendor'))
from utils.subphonetic_topology import SubphoneticTopologyLoss, decode_subphone_ids
for name, record in json.loads((HERE/'source_manifest.json').read_text()).items():
    path = HERE/'vendor'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['runtime_sha256'], path
assert Path(sys.modules['utils.subphonetic_topology'].__file__).resolve() == (HERE/'vendor/utils/subphonetic_topology.py').resolve()


def criterion(cost="ifmdd_standard_jsd"):
    return SubphoneticTopologyLoss(
        str(HERE/'mfa_label_encoder.txt'), context_phone_mode='mono',
        topology='s3_t2_star_star', split_silence=False, blank_index=0,
        loss_type='ottc', backend='torch', reduction='mean', ottc_cost=cost,
        ottc_mass_mode='state_uniform', ottc_soft_targets=False,
        ordinal_centroid_loss_weight=0., repeat_separator='optional_blank',
        silence_self_loops='none',
    )


def relative_lengths(lengths, maximum):
    # Follow IF-MDD's float relative-length API and its floor conversion intact.
    return lengths.float() / maximum


def frame_masses(raw, lengths):
    relative = relative_lengths(lengths, raw.shape[1])
    absolute = (relative * raw.shape[1]).long().clamp(1, raw.shape[1])
    mask = torch.arange(raw.shape[1], device=raw.device)[None] < absolute[:, None]
    return raw.masked_fill(~mask, -torch.inf).softmax(-1)


def ottc_objective(loss_module, logits, lengths, tokens, token_lengths, mass_logits=None):
    padded = pad_flat_targets(tokens, token_lengths)
    if bool(((tokens < 3) | (tokens >= 105)).any()) or bool((token_lengths <= 0).any()):
        raise ValueError('MFA OTTC targets must be nonempty and exclude controls/blank')
    masses = None if mass_logits is None else frame_masses(mass_logits.float(), lengths)
    return loss_module(
        logits.log_softmax(-1), padded,
        relative_lengths(lengths, logits.shape[1]),
        relative_lengths(token_lengths, padded.shape[1]),
        logits=logits, weights_logits=masses,
    )


def decode(logits, lengths, loss_module=None):
    loss_module = criterion() if loss_module is None else loss_module
    state_ids = ctc_greedy_decode(logits, relative_lengths(lengths, logits.shape[1]), blank_id=0)
    inventory = {label: idx for label, idx in
                 (line.rsplit(' ', 1) for line in Path(json.loads((HERE/'reference.json').read_text())['inventory']['path']).read_text().splitlines())}
    hypotheses = [[int(inventory[p]) for p in decode_subphone_ids(ids, loss_module.labels, mode='phone').split()]
                  for ids in state_ids]
    return hypotheses


class TopologyOTTC(nn.Module):
    topology = 's3_t2_star_star'
    objective = 'ifmdd_standard_jsd_spottc'
    frame_shift_seconds = .01

    def __init__(self, tiny=False):
        super().__init__()
        self.loss_module = criterion()
        p = parameters(tiny)
        self.acoustic = native.get_model(p)
        dim = native.lookup(p, 'embed_dim')
        self.acoustic.encoder_embed = FrameProjection(p.feature_dim, dim)
        self.acoustic.encoder.output_downsampling_factor = 1
        self.acoustic.ctc_output = nn.Linear(dim, self.loss_module.output_neurons)
        self.weight_head = nn.Linear(dim, 1)

    def acoustic_logits(self, features, lengths, aux_loss_scale=0.):
        enc, out_lengths = self.acoustic.forward_encoder(features, lengths, aux_loss_scale=aux_loss_scale)
        assert enc.shape[1] == features.shape[1] and torch.equal(lengths, out_lengths)
        # The topology inventory already excludes special content labels.
        return self.acoustic.ctc_output(enc).float(), out_lengths, self.weight_head(enc).squeeze(-1).float()

    def forward(self, features, lengths, tokens, token_lengths, frame_targets=None, aux_loss_scale=0.):
        logits, out_lengths, weights = self.acoustic_logits(features, lengths, aux_loss_scale)
        with torch.autocast(features.device.type, enabled=False):
            loss = ottc_objective(self.loss_module, logits, out_lengths, tokens, token_lengths, weights)
        debug = self.loss_module.last_k2_debug
        return loss, dict(
            ottc_ifmdd_standard_jsd_sum=loss.detach(),
            ottc_ifmdd_standard_jsd_per_utterance=loss.detach()/features.shape[0],
            ottc_effective_frames=logits.new_tensor(int(debug['input_lens'].sum())),
            ottc_effective_target_tokens=logits.new_tensor(int(debug['target_abs_lens'].sum())),
            ottc_uniform_fallback=logits.new_tensor(float(debug['ottc_fallback'] != 'none')),
            ottc_frame_mass_max=debug['frame_weights_max'].max().to(logits.device),
        )
