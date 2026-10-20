"""Explicit data-exposure schedule for a larger-batch single-head CE recipe."""
import math


def settings(batch_size):
    if batch_size not in (64,128,256):raise ValueError('Supported batches: 64, 128, 256')
    per_epoch=math.ceil(4620/batch_size)
    return dict(batch_size=batch_size,steps_per_epoch=per_epoch,total_steps=40*per_epoch,
        warmup_steps=round(500*per_epoch/73),
        frozen_steps_for_regfrozen=round(2165*per_epoch/73),
        learning_rate=2e-4,epochs=40,bins_metric='crossentropy')


def suffix(batch_size):return f'runexp_ottc_fuse_phoneme_ce_wavlm_b{batch_size}_20261013'
