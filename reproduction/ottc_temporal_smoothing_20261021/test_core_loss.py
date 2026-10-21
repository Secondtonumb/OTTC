"""Replay the pre-edit OTTC loss and validate optional smoothing in ot_loss.py."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ottc.ops_loss import ot_loss

HERE = Path(__file__).resolve().parent
BASELINE = HERE / 'core_integration/ot_loss_before_temporal_smoothing.py'
COSTS = [{}, {'jsd': True}, {'jsd': 'standard'}, {'euclidian': True}]


def cpu_function(source):
    # Device-only test adaptation; neither the formula nor production file is
    # changed. Reuses the same established CPU allocation gate as model tests.
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'batched_ottc_loss_bucketized')
    function_source = ast.get_source_segment(source, node).replace(".to('cuda')", ".to('cpu')")
    namespace = dict(vars(ot_loss))
    exec(compile(function_source, 'cpu_test_allocations_only', 'exec'), namespace)
    return namespace['batched_ottc_loss_bucketized']


@pytest.fixture
def functions():
    meta = json.loads((HERE / 'core_integration/BASELINE.json').read_text())
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == meta['sha256']
    assert 'smooth_weight' not in BASELINE.read_text()
    return cpu_function(BASELINE.read_text()), cpu_function(inspect.getsource(ot_loss.batched_ottc_loss_bucketized))


def inputs():
    torch.manual_seed(15)
    logits = torch.randn(2, 7, 5, requires_grad=True)
    raw_mass = torch.randn(2, 7, requires_grad=True)
    valid = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
    alpha = raw_mass.softmax(-1).masked_fill(~valid, 0.)
    alpha = alpha / alpha.sum(-1, keepdim=True)
    targets = F.one_hot(torch.tensor([[1, 2, 3], [2, 3, 0]]), 5)
    target_mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    beta = target_mask / target_mask.sum(-1, keepdim=True)
    return logits, raw_mass, alpha, targets, target_mask, beta, valid


@pytest.mark.parametrize('cost', COSTS)
def test_default_off_matches_original_value_gradient_and_transport_exactly(functions, cost):
    before, current = functions
    x, raw_mass, a, y, bm, b, valid = inputs()
    original = before(x, y, a, b, bmask=bm, **cost)
    for options in [{}, dict(smooth_weight=0.)]:
        updated = current(x, y, a, b, bmask=bm, **cost, **options)
        assert len(updated) == len(original) == 4
        for actual, expected in zip(updated, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        old_grads = torch.autograd.grad(original[0], (x, raw_mass), retain_graph=True)
        new_grads = torch.autograd.grad(updated[0], (x, raw_mass), retain_graph=True)
        for actual, expected in zip(new_grads, old_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('cost', COSTS)
def test_enabled_adds_only_the_requested_penalty_and_preserves_transport(functions, cost):
    before, current = functions
    x, raw_mass, a, y, bm, b, valid = inputs()
    original = before(x, y, a, b, bmask=bm, **cost)
    updated = current(x, y, a, b, bmask=bm, smooth_weight=.25, smooth_tau=4., smooth_mask=valid, **cost)
    expected = original[0] + .25 * ot_loss.temporal_smoothing_loss(x, valid, reduction='sum')
    torch.testing.assert_close(updated[0], expected, rtol=0, atol=0)
    for actual, old in zip(updated[1:], original[1:]):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    new_grads = torch.autograd.grad(updated[0], (x, raw_mass), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, raw_mass), retain_graph=True)
    for actual, reference in zip(new_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    old_mass_gradient = torch.autograd.grad(original[0], raw_mass)[0]
    torch.testing.assert_close(new_grads[1], old_mass_gradient, rtol=0, atol=0)


def test_amask_fallback_is_equivalent_to_explicit_smoothing_mask(functions):
    _, current = functions
    x, _, a, y, bm, b, valid = inputs()
    fallback = current(x, y, a, b, amask=valid, bmask=bm, jsd=True, smooth_weight=.1)
    explicit = current(x, y, a, b, amask=valid, bmask=bm, jsd=True, smooth_weight=.1, smooth_mask=valid)
    for actual, expected in zip(fallback, explicit):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('weight', [-.1, float('inf'), float('nan')])
def test_invalid_weight_is_rejected(functions, weight):
    _, current = functions
    x, _, a, y, bm, b, valid = inputs()
    with pytest.raises(ValueError, match='smooth_weight'):
        current(x, y, a, b, bmask=bm, smooth_weight=weight, smooth_mask=valid)


def test_enabled_requires_actual_output_frame_mask(functions):
    _, current = functions
    x, _, a, y, bm, b, valid = inputs()
    with pytest.raises(ValueError, match='requires smooth_mask or amask'):
        current(x, y, a, b, bmask=bm, smooth_weight=.1)
    with pytest.raises(ValueError, match='output-frame shape'):
        current(x, y, a, b, bmask=bm, smooth_weight=.1, smooth_mask=valid[:, :-1])
