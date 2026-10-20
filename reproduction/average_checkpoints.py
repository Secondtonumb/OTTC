"""Model parameter averaging; never average optimizer, scheduler or RNG state."""
import hashlib
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model_state(path, mfa=False):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    return state['model'] if mfa else state


def average(paths, mfa=False):
    if len(paths) < 2 or len(set(map(str, paths))) != len(paths):
        raise ValueError('Need at least two distinct checkpoint files')
    total = None
    dtypes = {}
    latest_nonfloating = {}
    differing_nonfloating = set()
    for index, path in enumerate(paths):
        state = load_model_state(path, mfa)
        if total is not None and set(state) != set(total):
            raise ValueError('Checkpoint state keys differ')
        if total is None:
            total = {}
            for key, value in state.items():
                if not torch.is_tensor(value):
                    raise TypeError('Unexpected non-tensor model state: '+key)
                dtypes[key] = value.dtype
                total[key] = torch.zeros_like(value, dtype=torch.float64) if value.is_floating_point() else value.clone()
        for key, value in state.items():
            if value.shape != total[key].shape or value.dtype != dtypes[key]:
                raise ValueError('Shape/dtype differs for '+key)
            if value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite source tensor: '+key)
                total[key].add_(value)
            else:
                if index and not torch.equal(latest_nonfloating[key], value):
                    differing_nonfloating.add(key)
                latest_nonfloating[key] = value.clone()
        del state
    for key in total:
        total[key] = (total[key].div_(len(paths)).to(dtypes[key])
                      if total[key].is_floating_point() else latest_nonfloating[key])
    return total, {'floating_accumulation': 'float64, uniform arithmetic mean; cast to original dtype',
                   'nonfloating_buffers': 'copied from newest selected checkpoint',
                   'differing_nonfloating_buffers': sorted(differing_nonfloating),
                   'optimizer_scheduler_rng': 'excluded; averaged artifact is inference-only'}
