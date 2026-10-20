"""Framewise phone CE on a frozen upstream-JSD S3T2 model's existing state head."""
import torch
import sys
sys.path.insert(0, '/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd')
from spottc_jsd_model import MFAZapformer


def targets(rows, lengths, device):
    out = torch.full((len(rows), int(lengths.max())), -100, device=device, dtype=torch.long)
    for b, row in enumerate(rows):
        n = int(lengths[b]); duration = float(torch.tensor(row['duration'], dtype=torch.float32))
        for label, start, end in row['intervals']:
            assert 3 <= label <= 104
            # Historical code materializes times in float32 before rounding.
            start = float(torch.tensor(start, dtype=torch.float32))
            end = float(torch.tensor(end, dtype=torch.float32))
            if end <= start: continue
            first = int(torch.floor(torch.tensor(start/duration*n, dtype=torch.float32)))
            last = int(torch.ceil(torch.tensor(end/duration*n, dtype=torch.float32)))
            first = max(0, min(n-1, first)); last = max(first+1, min(n, last))
            out[b, first:last] = label
    return out


class FrameCE(MFAZapformer):
    def __init__(self):
        super().__init__()
        self.weight_head.requires_grad_(False)

    def forward(self, features, lengths, frame_targets):
        enc, out_lengths = self.acoustic.forward_encoder(features, lengths, aux_loss_scale=0.)
        assert enc.shape[1] == features.shape[1] and torch.equal(lengths, out_lengths)
        state = self.acoustic.ctc_output(enc).float()
        state = state.index_fill(-1, torch.arange(1,7,device=state.device), -1e4)
        logp = state.log_softmax(-1)
        phones = torch.cat([logp[...,:1], torch.logsumexp(logp[...,1:].reshape(*logp.shape[:-1],104,3),-1)],-1)
        # Blank remains in the state-softmax denominator, matching historical CE.
        value = torch.nn.functional.nll_loss(phones.reshape(-1,105), frame_targets.reshape(-1),
                                            ignore_index=-100, reduction='sum')
        return value
