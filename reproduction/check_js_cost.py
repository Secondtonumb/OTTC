"""Check standard JS values and gradients against independent SciPy/NumPy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT)]
import numpy as np
from scipy.spatial.distance import jensenshannon
import torch
from ottc.ops_loss.ot_loss import (
    jensen_shannon_divergence, jensen_shannon_divergence_standard,
    batched_ottc_loss_bucketized,
)


def main():
    p = torch.tensor([[.7, .2, .1], [.2, .5, .3]], dtype=torch.float64)
    q = torch.tensor([[1., 0., 0.], [.3, .4, .3]], dtype=torch.float64)
    logits = p.log().requires_grad_()
    actual = jensen_shannon_divergence_standard(logits, q).sum(-1)
    expected = torch.tensor([jensenshannon(a, b)**2 for a, b in zip(p.numpy(), q.numpy())])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual, jensen_shannon_divergence_standard(q.clamp_min(1e-100).log(), p).sum(-1))
    assert torch.autograd.gradcheck(lambda x: jensen_shannon_divergence_standard(x, q).sum(), (logits,))
    extreme = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], requires_grad=True)
    extreme_loss = jensen_shannon_divergence_standard(extreme, q.float()).sum(-1)
    assert torch.isfinite(extreme_loss).all() and (extreme_loss >= 0).all()
    assert (extreme_loss <= np.log(2)+1e-6).all()
    extreme_loss.sum().backward()
    assert torch.isfinite(extreme.grad).all()
    upstream = jensen_shannon_divergence(p, q).sum(-1)
    qs = q+(1-q)*1e-12-q*1e-12
    m = (p+qs)/2
    reverse = .5*(m*(m.log()-p.log())+m*(m.log()-qs.log())).sum(-1)
    torch.testing.assert_close(upstream, reverse)
    assert not torch.allclose(upstream, actual)
    print('PASS: standard JS matches SciPy, symmetry, gradcheck, finite extreme logits; upstream is reverse KL')
    if torch.cuda.is_available():
        x = torch.randn(2, 4, 3, device='cuda', requires_grad=True)
        y = torch.nn.functional.one_hot(torch.tensor([[0, 1, 2], [2, 0, 0]], device='cuda'), 3)
        a = torch.tensor([[.1, .2, .3, .4], [.25, .25, .25, .25]], device='cuda', requires_grad=True)
        b = torch.tensor([[.3, .3, .4], [.5, .5, 0.]], device='cuda')
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], device='cuda')
        for cost in [True, 'standard']:
            loss, ix, iy, h = batched_ottc_loss_bucketized(x, y, a, b, bmask=mask, jsd=cost)
            assert torch.isfinite(loss)
            loss.backward(retain_graph=True)
            assert torch.isfinite(x.grad).all() and torch.isfinite(a.grad).all()
            x.grad = None
            a.grad = None
        print('PASS: both OT JS branches have finite logits and transport-weight gradients with padded targets')


if __name__ == '__main__':
    main()
