import torch
import torch.nn.functional as F
from ottc.ops_loss.gw_transport import (
    GWOptions, neighborhood_sum, structural_gradient, structure, monotone_plan,
    asot_plan, objective_terms, ag_cost,
)


def test_prefix_matches_dense_and_gradient():
    torch.manual_seed(1)
    q = torch.rand(7, 4, dtype=torch.double, requires_grad=True)
    t = torch.arange(7)
    a = (((t[:, None] - t).abs() <= 2) & (t[:, None] != t)).double()
    b = 1 - torch.eye(4, dtype=torch.double)
    expected = .5 * 49 / a.sum() * (q * (a @ q @ b)).sum()
    torch.testing.assert_close(structure(q, 2), expected)
    grad, = torch.autograd.grad(expected, q)
    torch.testing.assert_close(structural_gradient(q, 2), grad)
    torch.testing.assert_close(neighborhood_sum(q, 2), a @ q)


def test_monotone_marginals_and_gradients():
    a = torch.tensor([.12, .21, .19, .48], dtype=torch.double, requires_grad=True)
    b = torch.tensor([.3, .25, .45], dtype=torch.double)
    q = monotone_plan(a, b)
    torch.testing.assert_close(q.sum(1), a)
    torch.testing.assert_close(q.sum(0), b)
    entries = torch.nonzero(q > 0)
    assert torch.all(entries[1:, 1] >= entries[:-1, 1])
    assert torch.autograd.gradcheck(lambda v: structure(monotone_plan(v.softmax(0), b), 1),
                                   (torch.tensor([.1, .8, -.2, .3], dtype=torch.double, requires_grad=True),))


def test_asot_objective_gradient_consistency():
    torch.manual_seed(2)
    q = torch.rand(5, 3, dtype=torch.double, requires_grad=True)
    c = torch.rand_like(q)
    b = torch.ones(3, dtype=torch.double) / 3
    opt = GWOptions(mode='asot', order_prior=0)
    actual, = torch.autograd.grad(objective_terms(q, c, b, opt)['total'], q)
    expected = c + opt.weight * structural_gradient(q, opt.window_frames)
    expected = expected + opt.column_kl * (q.sum(0).log() - b.log())[None, :] + opt.entropy * q.log()
    torch.testing.assert_close(actual, expected)


def test_asot_rows_and_unrolled_mass_cost_gradients():
    torch.manual_seed(5)
    c = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    z = torch.randn(5, dtype=torch.double, requires_grad=True)
    b = torch.ones(3, dtype=torch.double) / 3
    opt = GWOptions(mode='asot', iterations=3)
    q = asot_plan(c, z.softmax(0), b, opt)
    torch.testing.assert_close(q.sum(1), z.softmax(0))
    assert torch.autograd.gradcheck(lambda cc, zz: objective_terms(
        asot_plan(cc, zz.softmax(0), b, opt), cc, b, opt)['total'], (c, z))


def test_short_and_large_windows():
    q = torch.tensor([[.3, .7]], requires_grad=True)
    assert structure(q, 50).item() == 0
    structure(q, 50).backward()
    assert q.grad.eq(0).all()
    q = torch.rand(3, 2)
    torch.testing.assert_close(neighborhood_sum(q, 50), q.sum(0, keepdim=True) - q)


def test_aba_penalty_vs_contiguous():
    contiguous = F.one_hot(torch.tensor([0, 0, 0, 1, 1, 1]), 2).float() / 6
    fragmented = F.one_hot(torch.tensor([0, 1, 0, 1, 0, 1]), 2).float() / 6
    assert structure(fragmented, 1) > structure(contiguous, 1)


def test_original_ag_formula_reused():
    from ottc.ops_loss.ot_loss import jensen_shannon_divergence
    torch.manual_seed(7)
    x = torch.randn(6, 4)
    labels = torch.tensor([1, 2, 1])
    cost = ag_cost(x, labels)
    expected = torch.stack([jensen_shannon_divergence(x.softmax(-1), F.one_hot(y, 4).float()).sum(-1)
                            for y in labels], dim=-1)
    torch.testing.assert_close(cost, expected)
