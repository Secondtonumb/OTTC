"""Dispatch existing frame loss helpers without changing their definitions."""
from reproduction.timit_frame_ce_20261026.model import frame_loss as ce_loss
from reproduction.timit_frame_divergences_20261026.model import frame_loss as divergence_loss


def frame_loss(logits, targets, mask, cost):
    if cost == 'ce':
        return ce_loss(logits, targets, mask)
    return divergence_loss(logits, targets, mask, cost)
