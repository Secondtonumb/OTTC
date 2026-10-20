"""Thin adapter to the existing, archived OTTC reproduction implementation.

No JSD formula or transport algorithm is defined here. This imports the exact
source used for the previous single-head JSD reproduction and calls jsd=True.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[2] / 'provenance/js_source/ottc/ops_loss/ot_loss.py'
SOURCE_SHA256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
_spec = importlib.util.spec_from_file_location('mfa_jsd_original_ottc_loss', SOURCE)
original_ottc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original_ottc)


def acoustic_columns(logits, targets):
    """Exclude non-acoustic controls 1/2 from the loss, keeping external IDs.

Zapformer fills those logits with -1e4. Upstream jsd takes log(softmax),
which is undefined for these zero-probability controls. They are never targets.
Blank 0, speech phones 3..102 and acoustic silence 103/104 stay in the loss.
No prediction probability clipping or divergence modification is applied.
"""
    if logits.shape[-1] != 105 or targets.shape[-1] != 105:
        raise ValueError('This adapter requires the original MFA 105-ID phone inventory')
    if bool((targets[..., 1:3] != 0).any()):
        raise ValueError('MFA control IDs must not be acoustic targets')
    indices = torch.cat([torch.zeros(1, device=logits.device, dtype=torch.long),
                         torch.arange(3, 105, device=logits.device)])
    return logits.index_select(-1, indices), targets.index_select(-1, indices)


def batched_ottc_jsd_loss(logits, targets, frame_weights, target_weights,
                         frame_mask, target_mask):
    logits, targets = acoustic_columns(logits, targets)
    return original_ottc.batched_ottc_loss_bucketized(
        logits, targets, frame_weights, target_weights,
        amask=frame_mask, bmask=target_mask, jsd=True,
    )[0]
