"""Verify the proposed scatter adapter without enabling it in the trainer."""
import runtime
import hashlib
import json
from pathlib import Path

import torch
import _k2
from ifmdd_losses import topology_forward_sum_loss
from k2_determinism import enable, enable_dense_broadcast

HERE = Path(__file__).resolve().parent


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.autograd.set_multithreading_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1729)
    native_index_add = _k2.index_add
    enable_dense_broadcast()
    original = torch.randn(3, 23, 313, device='cuda').log_softmax(-1).detach()
    lengths = torch.tensor([23, 19, 13], device='cuda')
    tokens = torch.tensor([7, 12, 7, 12, 17, 7], device='cuda')
    token_lengths = torch.tensor([3, 2, 1], device='cuda')

    def gradient():
        logp = original.clone().requires_grad_()
        loss = topology_forward_sum_loss(logp, lengths, tokens, token_lengths, topology='s3_t2')
        grad, = torch.autograd.grad(loss, logp)
        return loss.detach(), grad

    native = [gradient() for _ in range(12)]
    enable()
    deterministic = [gradient() for _ in range(12)]
    loss, grad = deterministic[0]
    for l, g in deterministic:
        assert torch.equal(l, loss) and torch.equal(g, grad)
    for l, g in native:
        assert torch.equal(l, loss), 'Forward loss changed'
        torch.testing.assert_close(g, grad, rtol=1e-5, atol=1e-7)
    assert bool(torch.isfinite(grad).all())
    assert float(grad.abs().sum()) > 0

    cases = []
    for dtype in (torch.float32, torch.float64, torch.int32):
        for columns in (None, 4):
            index = torch.tensor([2, -1, 1, 2, 0, -1, 2], device='cuda', dtype=torch.int32)
            shape = (len(index),) if columns is None else (len(index), columns)
            value = torch.arange(int(torch.tensor(shape).prod()), device='cuda', dtype=dtype).reshape(shape)
            out_shape = (4,) if columns is None else (4, columns)
            initial = torch.full(out_shape, 3, device='cuda', dtype=dtype)
            expected = initial.clone()
            native_index_add(index, value, expected)
            actual = initial.clone()
            _k2.index_add(index, value, actual)
            assert torch.equal(expected, actual), (dtype, shape)
            cases.append(dict(dtype=str(dtype),value_shape=list(shape),ignored_negative_indices=True,
                              duplicate_indices=True,nonzero_initial_output=True,exact_match=True))

    results = dict(torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                   native_gradient_distinct_hashes=len({digest(g) for _,g in native}),
                   deterministic_gradient_distinct_hashes=len({digest(g) for _,g in deterministic}),
                   repeats_per_mode=12,loss=float(loss),loss_identical=True,
                   gradient_native_max_abs_difference=max(float((g-grad).abs().max()) for _,g in native),
                   gradient_comparison='rtol=1e-5 atol=1e-7',index_add_semantics=cases,
                   multi_utterance_native_constructor='explicit terminal-row broadcast; native construction unchanged',
                   trainer_default_changed=False)
    output=HERE/'diagnostics_k2/gradient_equivalence.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__ == '__main__': main()
