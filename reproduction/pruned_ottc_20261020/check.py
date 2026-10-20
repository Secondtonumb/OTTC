"""Local CPU/CUDA checks against frozen upstream functions; no W&B uploads."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.pruned_ottc_20261020.loss import pruned_ottc_loss, reduced_log_softmax, remap_targets
from reproduction.pruned_ottc_20261020.model import PrunedOTTCMixin, PrunedTrainingOutput
from reproduction.pruned_ottc_20261020 import loss as loss_module

ARCHIVE = ROOT / 'reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py'
BASE_MODEL = ROOT / 'ottc/models/sequence/wav2vec2_ottc.py'


def upstream_functions(device='cpu'):
    class DeviceOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 'cuda':
                return ast.copy_location(ast.Constant(device), node)
            return node

    tree = ast.parse(ARCHIVE.read_text())
    names = {'jensen_shannon_divergence', 'batched_ottc_loss_bucketized'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    module = ast.fix_missing_locations(DeviceOnly().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = dict(torch=torch, F=functional, batched_bucketize=torch.vmap(torch.bucketize))
    exec(compile(module, str(ARCHIVE), 'exec'), namespace)
    return namespace['jensen_shannon_divergence'], namespace['batched_ottc_loss_bucketized']


def check_mapping():
    targets = torch.tensor([[17, 42, 17], [42, 9, -100]])
    vocabulary = remap_targets(targets, 64)
    assert vocabulary.selected.tolist() == [9, 17, 42]
    assert vocabulary.targets.tolist() == [[1, 2, 1], [2, 0, -100]]
    valid = targets != -100
    assert torch.equal(vocabulary.selected[vocabulary.targets[valid]], targets[valid])
    for invalid in (torch.tensor([[0]]), torch.tensor([[-100]]), torch.tensor([[64]]),
                    torch.tensor([[9, -100, 17]]), torch.tensor([[-2]])):
        try:
            remap_targets(invalid, 64)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid targets accepted')


def check_case(dtype, cost, chunk_size, with_bias=True, frozen_head=False, device='cpu'):
    torch.manual_seed(42)
    jsd, original_loss = upstream_functions(device)
    hidden = (torch.randn(2, 5, 4, dtype=dtype, device=device) * .3).requires_grad_()
    weight = torch.randn(11, 4, dtype=dtype, device=device, requires_grad=not frozen_head)
    bias = torch.randn(11, dtype=dtype, device=device, requires_grad=not frozen_head) if with_bias else None
    mass_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    frame_logits = torch.randn(2, 5, dtype=mass_dtype, device=device, requires_grad=True)
    target_logits = torch.randn(2, 3, dtype=mass_dtype, device=device, requires_grad=True)
    frame_mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2], device=device)
    targets = torch.tensor([[3, 7, 3], [7, 9, -100]], device=device)
    frame_mass = frame_logits.masked_fill(~frame_mask, -torch.inf).softmax(-1)
    target_mass = target_logits.masked_fill(targets == -100, -torch.inf).softmax(-1)
    output = pruned_ottc_loss(hidden, weight, bias, targets, frame_mass, input_mask=frame_mask,
        target_mass=target_mass, cost=cost, chunk_size=chunk_size, jsd_function=jsd)
    working = mass_dtype
    logits = functional.linear(hidden.to(working), weight.to(working),
                               None if bias is None else bias.to(working))
    one_hot = functional.one_hot(targets.clamp_min(0), len(weight))
    expected, frame_indices, target_indices, transport = original_loss(
        logits, one_hot, frame_mass, target_mass, amask=frame_mask, bmask=targets != -100,
        jsd=cost == 'jsd')
    torch.testing.assert_close(output.transport_weights, transport)
    torch.testing.assert_close(output.frame_indices, frame_indices.long())
    torch.testing.assert_close(output.target_indices, target_indices.long())
    parameters = [value for value in (hidden, weight, bias, frame_logits, target_logits)
                  if value is not None and value.requires_grad]
    actual_gradients = torch.autograd.grad(output.loss, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    tolerance = 2e-10 if dtype == torch.float64 else (2e-5 if dtype == torch.float32 else .025)
    torch.testing.assert_close(output.loss, expected, atol=tolerance, rtol=tolerance)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    assert not bool(actual_gradients[0][~frame_mask].any())
    assert bool(actual_gradients[0][frame_mask].abs().sum() > 0)
    return dict(dtype=str(dtype), cost=cost, chunk_size=chunk_size, bias=with_bias,
                frozen_head=frozen_head, loss_error=float((output.loss - expected).abs().detach()),
                max_gradient_error=max(float((actual - reference).abs().max())
                    for actual, reference in zip(actual_gradients, expected_gradients)))


def check_log_softmax():
    hidden = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 2, dtype=torch.float64, requires_grad=True)
    selected = torch.tensor([6, 1, 4])
    assert torch.autograd.gradcheck(
        lambda features, projection: reduced_log_softmax(features, projection, None, selected, chunk_size=2),
        (hidden, weight))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        try:
            reduced_log_softmax(hidden, weight, None, selected)
        except RuntimeError:
            pass
        else:
            raise AssertionError('autocast must be explicitly disabled')


def check_model(device='cpu'):
    jsd, original_loss = upstream_functions(device)
    tree = ast.parse(BASE_MODEL.read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'wvForOTTC')
    names = {'forward', 'regular_forward', 'regular_forward_and_frozen'}
    methods = [node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *methods], type_ignores=[])
    namespace = dict(torch=torch, F=functional, copy=copy, CausalLMOutput=PrunedTrainingOutput,
                     batched_ottc_loss_bucketized=original_loss)
    exec(compile(ast.fix_missing_locations(module), str(BASE_MODEL), 'exec'), namespace)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4)

        def forward(self, input_values, **kwargs):
            return (self.projection(input_values),)

    class TinyOTTC(torch.nn.Module):
        regular_forward = namespace['regular_forward']
        regular_forward_and_frozen = namespace['regular_forward_and_frozen']
        forward = namespace['forward']

        def __init__(self, cost, mode):
            super().__init__()
            self.config = SimpleNamespace(use_return_dict=True, vocab_size=11, pad_token_id=0)
            self.wave2vec2_config = SimpleNamespace()
            self.wav2vec2 = Encoder()
            self.wav2vec2_freeze = copy.deepcopy(self.wav2vec2).requires_grad_(False)
            self.lm_head = torch.nn.Linear(4, 11)
            self.lm_weight = torch.nn.Linear(4, 1)
            self.dropout = torch.nn.Identity()
            self.forward_type = mode
            self.frozen_steps_for_regfrozen = 2
            self.oracle_weights_model_bool = False
            self.euclidian, self.jsd = False, cost == 'jsd'
            self.step, self.count = -1, 0

        def _get_feat_extract_output_lengths(self, config, lengths):
            return lengths

        def create_attention_mask_from_input_sequence(self, lengths):
            return torch.arange(int(lengths.max()), device=lengths.device)[None, :] < lengths[:, None]

    class TinyPrunedOTTC(PrunedOTTCMixin, TinyOTTC):
        pass

    results = []
    for cost in ('ce', 'jsd'):
        for mode in ('regular', 'regfrozen'):
            torch.manual_seed(71)
            dense = TinyOTTC(cost, mode).to(device)
            pruned = TinyPrunedOTTC(cost, mode, pruned_chunk_size=3, pruned_jsd_function=jsd).to(device)
            pruned.load_state_dict(dense.state_dict(), strict=True)
            samples = torch.randn(2, 5, 4, device=device)
            mask = torch.tensor([[1] * 5, [1] * 3 + [0] * 2], device=device)
            label_batches = (torch.tensor([[3, 7, 3], [7, 9, -100]], device=device),
                             torch.tensor([[2, 4, -100], [4, 5, 4]], device=device))
            dense_optimizer = torch.optim.SGD(dense.parameters(), lr=.01)
            pruned_optimizer = torch.optim.SGD(pruned.parameters(), lr=.01)
            for step in range(3):
                labels = label_batches[step % 2]
                dense.step = pruned.step = step - 1
                dense_optimizer.zero_grad(set_to_none=True)
                pruned_optimizer.zero_grad(set_to_none=True)
                projected_widths = []
                original_projection = loss_module._projection

                def record_projection(features, weight, bias, dtype):
                    projected_widths.append(len(weight))
                    return original_projection(features, weight, bias, dtype)

                with mock.patch.object(loss_module, '_projection', side_effect=record_projection):
                    actual = pruned(samples, attention_mask=mask, labels=labels)
                    actual.loss.backward()
                expected = dense(samples, attention_mask=mask, labels=labels)
                expected.loss.backward()
                torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-5)
                assert actual.logits is None and not pruned.lm_head.deferred
                assert projected_widths and max(projected_widths) <= 3
                assert torch.equal(pruned.last_pruned_vocabulary.selected, labels[labels >= 0].unique())
                dense_parameters = dict(dense.named_parameters())
                for name, parameter in pruned.named_parameters():
                    reference = dense_parameters[name]
                    if reference.grad is None:
                        assert parameter.grad is None
                    else:
                        torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-5, atol=2e-5)
                dense_optimizer.step()
                pruned_optimizer.step()
            pruned.eval()
            dense.eval()
            with torch.no_grad():
                actual = pruned(samples, attention_mask=mask)
                expected = dense(samples, attention_mask=mask)
                torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-5)
                assert actual.logits.shape == (2, 5, 11)
                actual = pruned(samples, attention_mask=mask, labels=labels)
                expected = dense(samples, attention_mask=mask, labels=labels)
                torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-5)
            restored = TinyOTTC(cost, mode).to(device)
            restored.load_state_dict(pruned.state_dict(), strict=True)
            results.append(dict(cost=cost, forward_type=mode, optimizer_steps=3,
                changing_batch_vocabulary=True, full_projection_during_training=False,
                inference='unchanged full-vocabulary base forward', state_dict_compatible=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    check_mapping()
    check_log_softmax()
    results = [check_case(dtype, cost, chunk, device=args.device) for dtype in (torch.float64, torch.float32, torch.bfloat16)
               for cost in ('ce', 'jsd') for chunk in (1, 4, 4096)]
    results.extend(check_case(torch.float64, cost, 3, with_bias=bias, frozen_head=frozen, device=args.device)
                   for cost in ('ce', 'jsd') for bias, frozen in ((False, False), (True, True)))
    model_results = check_model(args.device)
    report = dict(passed=True, torch_version=torch.__version__, device=args.device, seed=42,
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [ARCHIVE, BASE_MODEL, Path(__file__), Path(__file__).with_name('loss.py'),
                                    Path(__file__).with_name('model.py')]},
        reference_adaptation=f'AST loads original functions; device literals set to {args.device}.',
        evaluation_uploads=False, cases=results, model_cases=model_results)
    output = args.output or Path(__file__).with_name('checks') / (args.device + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(results)} dense-reference cases, batch mapping, masks, original transport, gradcheck and autocast guard.')
    print('PASS: four original-forward adapter cases, changing batch vocabulary, joint/frozen gradients, optimizer steps and inference.')
    print(f'Local report: {output}')


if __name__ == '__main__':
    main()