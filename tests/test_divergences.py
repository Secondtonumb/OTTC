"""Numerical definitions, derivatives, and OT integration; no data/model downloads."""
import ast
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ottc.ops_loss import ot_loss


def test_kl_directions_and_jeffreys_against_torch_distributions():
    p = torch.tensor([[.1, .3, .6], [.7, .2, .1]], dtype=torch.float64)
    q = torch.tensor([[.2, .5, .3], [.2, .4, .4]], dtype=torch.float64)
    logits = p.log().requires_grad_()
    forward = ot_loss.kullback_leibler_divergence(logits, q).sum(-1)
    reverse = ot_loss.kullback_leibler_divergence(logits, q, reverse=True).sum(-1)
    kl = torch.distributions.kl_divergence
    cat = torch.distributions.Categorical
    torch.testing.assert_close(forward, kl(cat(probs=p), cat(probs=q)))
    torch.testing.assert_close(reverse, kl(cat(probs=q), cat(probs=p)))
    assert not torch.allclose(forward, reverse)
    j = ot_loss.jeffreys_divergence(logits, q).sum(-1)
    torch.testing.assert_close(j, forward + reverse)
    torch.testing.assert_close(j, ot_loss.jeffreys_divergence(q.log(), p).sum(-1))
    # Independent finite-difference checks for the actual logits derivatives.
    for fn in (ot_loss.kullback_leibler_divergence, ot_loss.jeffreys_divergence):
        assert torch.autograd.gradcheck(lambda z: fn(z, q), (logits,))


def test_zero_at_equality_and_nonnegative():
    p = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    for fn in (ot_loss.kullback_leibler_divergence, ot_loss.jeffreys_divergence):
        torch.testing.assert_close(fn(p.log(), p).sum(-1), torch.zeros(1, dtype=p.dtype), atol=1e-14, rtol=0)
        costs = fn(torch.randn(20, 3, dtype=p.dtype), p.expand(20, -1)).sum(-1)
        assert (costs >= 0).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('metric', ['kl', 'kl_reverse', 'jdiv'])
def test_one_hot_and_extreme_logits_are_finite(dtype, metric):
    logits = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], dtype=dtype, requires_grad=True)
    q = F.one_hot(torch.tensor([1, 0]), 3).to(dtype)
    cost = ot_loss._divergence_cost(logits, q, metric)
    assert torch.isfinite(cost).all()
    assert (cost > 0).all()
    grad, = torch.autograd.grad(cost.sum(), logits)
    assert torch.isfinite(grad).all()
    assert cost.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)


def test_normalized_epsilon_target_and_direction():
    logits = torch.tensor([[.2, -.3, .8]], dtype=torch.float64)
    q = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    eps = .01
    qe = q.clamp_min(eps)
    qe = qe / qe.sum(-1, keepdim=True)
    p = logits.softmax(-1)
    expected = p * (p.log() - qe.log())
    torch.testing.assert_close(ot_loss.kullback_leibler_divergence(logits, q, eps=eps), expected)
    torch.testing.assert_close(ot_loss.jeffreys_divergence(logits, q, eps=eps), (p-qe) * (p.log()-qe.log()))


@pytest.mark.parametrize('eps', [0., -1., 1., float('nan'), float('inf')])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match='eps'):
        ot_loss.jeffreys_divergence(torch.zeros(1, 3), torch.ones(1, 3)/3, eps=eps)


def cpu_loss(fn):
    # Production OT allocates CUDA tensors unconditionally. Adapt device
    # allocations only, following the existing temporal-smoothing regression.
    source = inspect.getsource(fn).replace(".to('cuda')", ".to('cpu')")
    source = source.replace("device = 'cuda'", "device = 'cpu'")
    namespace = dict(vars(ot_loss))
    exec(compile(source, '<cpu allocations only>', 'exec'), namespace)
    return namespace[fn.__name__]


@pytest.mark.parametrize('metric', ['ag', 'kl', 'kl_reverse', 'jdiv'])
def test_transport_weighted_cost_and_gradients(metric):
    torch.manual_seed(34)
    x = torch.randn(1, 3, 4, requires_grad=True)
    raw_mass = torch.randn(1, 3, requires_grad=True)
    a = raw_mass.softmax(-1)
    y = F.one_hot(torch.tensor([[1, 2]]), 4).float()
    b = torch.tensor([[.4, .6]])
    bucket = cpu_loss(ot_loss.batched_ottc_loss_bucketized)
    concat = cpu_loss(ot_loss.batched_ottc_loss_concatenated)
    result = bucket(x, y, a, b, divergence=metric)
    cost, ix, iy, weights = result
    pair_cost = ot_loss._divergence_cost(x[0, ix[0]], y[0, iy[0]], metric)
    torch.testing.assert_close(cost, (weights[0] * pair_cost).sum())
    torch.testing.assert_close(cost, concat(x, y, a, b, divergence=metric))
    for grad in torch.autograd.grad(cost, (x, raw_mass), retain_graph=True):
        assert torch.isfinite(grad).all()
    if metric == 'ag':
        old = bucket(x, y, a, b, jsd=True)
        for actual, expected in zip(result, old):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(torch.autograd.grad(cost, (x, raw_mass), retain_graph=True),
                                    torch.autograd.grad(old[0], (x, raw_mass))):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        valid = torch.ones(1, 3, dtype=torch.bool)
        smoothed = bucket(x, y, a, b, divergence=metric, smooth_weight=.2, smooth_mask=valid)
        torch.testing.assert_close(smoothed[0], cost + .2 * ot_loss.temporal_smoothing_loss(x, valid, reduction='sum'))


@pytest.mark.parametrize('fn', [ot_loss.batched_ottc_loss_bucketized, ot_loss.batched_ottc_loss_concatenated])
def test_invalid_metric_and_conflicts_fail_before_transport(fn):
    for options in [dict(divergence='unknown'), dict(divergence='kl', jsd=True),
                    dict(divergence='jdiv', euclidian=True)]:
        with pytest.raises(ValueError, match='divergence'):
            fn(None, None, None, None, **options)


@pytest.mark.parametrize('metric,expected_jsd,expected_div', [
    ('crossentropy', False, None), ('jsd', True, None),
    ('jsd_standard', 'standard', None), ('ag', False, 'ag'),
    ('kl', False, 'kl'), ('kl_reverse', False, 'kl_reverse'), ('jdiv', False, 'jdiv'),
])
def test_model_metric_selection_without_loading_encoder(metric, expected_jsd, expected_div):
    # Execute the real __init__ selector, stopping immediately before pretrained
    # encoder loading, so this regression is independent of transformers/caches.
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'ottc/models/sequence/wav2vec2_ottc.py'
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'wvForOTTC')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.jsd')
    end = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.wav2vec2')
    selector = ast.Module(body=init.body[start:end], type_ignores=[])
    config = SimpleNamespace(bins_metric=metric)
    model = SimpleNamespace(config=config)
    exec(compile(selector, str(path), 'exec'), dict(self=model, config=config))
    assert model.jsd == expected_jsd
    assert model.divergence == expected_div
    assert not model.euclidian
