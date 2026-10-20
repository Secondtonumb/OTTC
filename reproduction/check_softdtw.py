"""Verify upstream Soft-DTW CE loss and gradient against PyTorch DP."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT)]
import torch
import torch.nn.functional as F
from ottc.ops_loss.soft_dtw_loss import SoftDTW


def reference(distance, gamma):
    zero = distance.new_zeros(())
    inf = distance.new_tensor(float('inf'))
    previous = [zero]+[inf]*distance.shape[1]
    for row in distance:
        current = [inf]
        for j, cost in enumerate(row, 1):
            choices = torch.stack([previous[j-1], previous[j], current[j-1]])
            current.append(cost-gamma*torch.logsumexp(-choices/gamma, dim=0))
        previous = current
    return previous[-1]


def main():
    torch.manual_seed(42)
    for gamma in [.01, .1, 1.]:
        logits = torch.randn(5, 4, requires_grad=True)
        labels = F.one_hot(torch.tensor([0, 1, 1]), 4)
        distance = -(labels.float() @ F.log_softmax(logits, -1).T).T
        expected = reference(distance, gamma)
        actual = SoftDTW(gamma=gamma, normalize=False, cost='ce')(logits, labels)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
        reference_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, reference_grad, rtol=2e-4, atol=2e-5)
        assert torch.isfinite(actual_grad).all()
    print('PASS: upstream CE Soft-DTW values and logits gradients match independent PyTorch DP, gamma .01/.1/1', flush=True)
    if torch.cuda.is_available():
        # CPU dynamic programming and repeated GPU/CPU transfers are upstream.
        timings = []
        for n, m in [(200, 40), (400, 80)]:
            x = torch.randn(n, 42, device='cuda', requires_grad=True)
            y = F.one_hot(torch.arange(m, device='cuda')%40, 42)
            torch.cuda.synchronize()
            start = time.monotonic()
            loss = SoftDTW(gamma=.1, cost='ce')(x, y)
            loss.backward()
            torch.cuda.synchronize()
            assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
            timings.append(dict(frames=n, phones=m, seconds=time.monotonic()-start))
        print('PASS: finite GPU CE loss/gradient', json.dumps(timings), flush=True)


if __name__ == '__main__':
    main()
