"""IF-MDD content CTC criterion on the frozen native Zapformer encoder."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "zapformer_s3t2ss_native_20261012"
sys.path.insert(0, str(NATIVE))
import runtime
import torch
from torch import nn
import train as native
from task_model import parameters, FrameProjection
from ifmdd_losses import pad_flat_targets
from speechbrain.nnet.losses import ctc_loss
from speechbrain.decoders.ctc import ctc_greedy_decode


def ctc_objective(logits, lengths, tokens, token_lengths, reduction="mean"):
    """Call the same SpeechBrain criterion selected by the IF-MDD FiLM recipe.

    Targets keep their original MFA IDs and repeated occurrences. Blank is an
    acoustic output, never an inserted transcript target. Relative lengths are
    converted back with SpeechBrain's rounding rule, including padded batches.
    """
    if logits.shape[-1] != 105:
        raise ValueError("This migration requires the existing MFA 105-class inventory")
    if int(token_lengths.sum()) != tokens.numel() or bool((token_lengths <= 0).any()):
        raise ValueError("Empty or inconsistent targets")
    if bool(((tokens < 3) | (tokens >= 105)).any()):
        raise ValueError("CTC targets must exclude blank and control IDs")
    padded = pad_flat_targets(tokens, token_lengths)
    # Preserve the native recipe's deterministic policy. CUDA CTC backward is
    # nondeterministic in this PyTorch build; the differentiable CPU copy keeps
    # the original SpeechBrain/PyTorch algorithm, without changing its formula.
    scores = logits.float().log_softmax(-1).cpu()
    return ctc_loss(
        scores, padded.cpu(),
        lengths.float().cpu() / logits.shape[1],
        token_lengths.float().cpu() / padded.shape[1],
        blank_index=0, reduction=reduction,
    ).to(logits.device)


def decode(logits, lengths):
    """The IF-MDD content-head decoder: merge repeats, then remove blank."""
    return ctc_greedy_decode(logits, lengths.float() / logits.shape[1], blank_id=0)


class PhoneCTC(nn.Module):
    topology = "standard_ctc"
    objective = "ctc"
    frame_shift_seconds = 0.01

    def __init__(self, tiny=False):
        super().__init__()
        p = parameters(tiny)
        self.acoustic = native.get_model(p)
        dim = native.lookup(p, "embed_dim")
        self.acoustic.encoder_embed = FrameProjection(p.feature_dim, dim)
        self.acoustic.encoder.output_downsampling_factor = 1
        # IF-MDD ctc_lin wraps nn.Linear, with a single class per phone/token.
        self.acoustic.ctc_output = nn.Linear(dim, 105)

    def acoustic_logits(self, features, lengths, aux_loss_scale=0.0):
        enc, out_lengths = self.acoustic.forward_encoder(
            features, lengths, aux_loss_scale=aux_loss_scale
        )
        assert enc.shape[1] == features.shape[1] and torch.equal(lengths, out_lengths)
        # Preserve the existing MFA convention for unavailable SOS/UNK outputs.
        logits = self.acoustic.ctc_output(enc).float()
        logits = logits.index_fill(-1, torch.tensor([1, 2], device=logits.device), -1e4)
        return logits, out_lengths

    def forward(self, features, lengths, tokens, token_lengths,
                frame_targets=None, aux_loss_scale=0.0):
        logits, out_lengths = self.acoustic_logits(features, lengths, aux_loss_scale)
        with torch.autocast(features.device.type, enabled=False):
            per_target = ctc_objective(logits, out_lengths, tokens, token_lengths, "batch")
            loss = per_target.mean()
            raw = per_target * token_lengths
        repeats = sum(int((seq[1:] == seq[:-1]).sum())
                      for seq in torch.split(tokens, token_lengths.tolist()))
        return loss, dict(
            ctc_target_normalized_mean=loss.detach(),
            ctc_raw_nll_sum=raw.sum().detach(),
            ctc_target_normalized_sum=per_target.sum().detach(),
            ctc_adjacent_target_repeats=logits.new_tensor(repeats),
        )
