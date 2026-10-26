"""Opt-in deterministic accumulation for k2's dense-FSA backward scatter.

The native k2 graph construction, forward scores and arc posteriors are kept.
Only index_add's CUDA accumulation order changes; -1 indices remain ignored.
This is a phone-task dependency adapter, not a Zapformer or OTTC loss rewrite.
"""
import torch
from torch.overrides import TorchFunctionMode


class ExplicitTerminalRows(TorchFunctionMode):
    """Materialize native DenseFsaVec's unique terminal-row broadcast.

    PyTorch 2.6's deterministic index_put rejects this implicit broadcast for
    multi-utterance batches. Expanding before autograd records the assignment
    preserves both the native construction and the native backward graph.
    This mode is scoped to construction, not to the network or optimizer.
    """
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.Tensor.__setitem__:
            target, indices, values = args
            if (target.ndim == 2 and isinstance(indices, list) and all(isinstance(i, int) for i in indices)
                    and torch.is_tensor(values) and values.ndim <= 1
                    and values.numel() in (1, target.shape[1])):
                values = values.expand(len(indices), target.shape[1]).contiguous()
                args = (target, indices, values)
        return func(*args, **kwargs)


def enable_dense_broadcast():
    import k2
    if getattr(k2.DenseFsaVec, '_deterministic_broadcast_adapter', False):
        return
    original = k2.DenseFsaVec

    class DeterministicDenseFsaVec(original):
        _deterministic_broadcast_adapter = True

        def __init__(self, log_probs, supervision_segments, allow_truncate=0):
            if torch.are_deterministic_algorithms_enabled() and log_probs.device.type == 'cuda':
                with ExplicitTerminalRows():
                    super().__init__(log_probs, supervision_segments, allow_truncate)
            else:
                super().__init__(log_probs, supervision_segments, allow_truncate)

    k2.DenseFsaVec = DeterministicDenseFsaVec


def enable():
    import _k2
    enable_dense_broadcast()
    if getattr(_k2.index_add, '_deterministic_adapter', False):
        return
    original = _k2.index_add

    def index_add(index, value, in_out):
        if not torch.are_deterministic_algorithms_enabled() or index.device.type != 'cuda':
            return original(index, value, in_out)
        valid = index >= 0
        in_out.index_add_(0, index[valid].long(), value[valid])

    index_add._deterministic_adapter = True
    _k2.index_add = index_add
