"""Numerical/gradient tests and an actual tiny OTTC integration on CPU.

Evaluation-only/local. No W&B initialization or dataset download.
"""
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import transformers

transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import Wav2Vec2Config, Wav2Vec2Model, TrainingArguments

import ottc.models.sequence.wav2vec2_ottc as upstream
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_smoothing import TemporalSmoothingMixin
from ottc.ops_loss.temporal_smoothing import temporal_smoothing_loss as smooth
from ottc.utils.temporal_smoothing_trainer import TemporalSmoothingTrainer


def test_original_mstcn_value_and_stop_gradient():
    torch.manual_seed(3)
    x = torch.randn(2, 6, 4, requires_grad=True)
    # Original author's [B,C,T] expression, with no padding and no 0.15 weight.
    p = x.transpose(1, 2)
    original = F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1),
        F.log_softmax(p.detach()[:, :, :-1], dim=1), reduction='none').clamp(0, 16).mean()
    adapted = smooth(x, torch.ones(2, 6, dtype=torch.bool))
    torch.testing.assert_close(adapted, original)
    ga, go = [torch.autograd.grad(loss, x, retain_graph=True)[0] for loss in (adapted, original)]
    torch.testing.assert_close(ga, go)
    assert torch.count_nonzero(ga[:, 0]) == 0  # Previous/reference branch detached.
    assert torch.count_nonzero(ga[:, -1]) > 0


def test_padding_holes_nonfinite_padding_and_normalization():
    torch.manual_seed(4)
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    x = torch.randn(2, 5, 3).masked_fill(~mask[:, :, None], float('nan')).requires_grad_()
    # Independent enumeration includes (0,1), (3,4), (0,1), (1,2) only.
    terms = []
    for b in range(2):
        for t in range(1, 5):
            if mask[b, t] and mask[b, t-1]:
                terms.append((F.log_softmax(x[b, t], -1) -
                              F.log_softmax(x[b, t-1], -1).detach()).square().clamp(max=16))
    expected = torch.stack(terms).mean()
    actual = smooth(x, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~mask]) == 0
    # Adding arbitrary padded time must change neither loss nor normalization.
    padded = F.pad(x.detach().nan_to_num(), (0, 0, 0, 7), value=999.)
    torch.testing.assert_close(smooth(padded, F.pad(mask, (0, 7))), actual.detach())


@pytest.mark.parametrize('shape', [(2, 0, 4), (2, 1, 4), (2, 5, 4)])
def test_no_valid_pairs_has_zero_finite_gradient(shape):
    x = torch.randn(shape, requires_grad=True)
    mask = torch.zeros(shape[:2], dtype=torch.bool)
    for reduction in ['mean', 'sum']:
        loss = smooth(x, mask, reduction=reduction)
        assert loss.item() == 0.
        grad = torch.autograd.grad(loss, x)[0]
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) == 0


def test_clipping_stops_saturated_gradient():
    x = torch.tensor([[[-50., 50.], [50., -50.]]], requires_grad=True)
    loss = smooth(x, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == 16.
    loss.backward()
    assert torch.count_nonzero(x.grad) == 0


def test_microbatch_additivity_and_gradients():
    torch.manual_seed(5)
    x = torch.randn(4, 7, 5, requires_grad=True)
    mask = torch.arange(7)[None, :] < torch.tensor([7, 2, 5, 1])[:, None]
    full = smooth(x, mask, reduction='sum')
    split = sum(smooth(x[a:b], mask[a:b], reduction='sum') for a, b in [(0, 1), (1, 3), (3, 4)])
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(torch.autograd.grad(full, x, retain_graph=True)[0],
                               torch.autograd.grad(split, x)[0])


def test_mixed_precision_accumulates_in_fp32():
    x = torch.randn(2, 4, 6, dtype=torch.bfloat16, requires_grad=True)
    loss = smooth(x, torch.ones(2, 4, dtype=torch.bool))
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_inputs_are_rejected():
    x, mask = torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool)
    for tau in [-1., 0., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            smooth(x, mask, tau=tau)
    with pytest.raises(ValueError):
        smooth(x, mask[:, :-1])
    with pytest.raises(ValueError):
        smooth(x, mask, reduction='unsupported')


class DeviceOTTC(upstream.wvForOTTC):
    def create_attention_mask_from_input_sequence(self, lengths):
        return (torch.arange(1, int(lengths.max()) + 1, device=lengths.device)[None] <= lengths[:, None]).float()


class TinySmoothedOTTC(TemporalSmoothingMixin, DeviceOTTC):
    pass


@contextmanager
def cpu_loss_allocations():
    # Test-only device adaptation of literal CUDA allocations, following the
    # repository's established CPU gate. No formula or source file is edited.
    original = upstream.batched_ottc_loss_bucketized
    namespace = dict(original.__globals__)
    source = inspect.getsource(original).replace(".to('cuda')", ".to('cpu')")
    exec(compile(source, 'cpu_test_allocations_only', 'exec'), namespace)
    with patch.object(upstream, 'batched_ottc_loss_bucketized', namespace[original.__name__]):
        yield


@pytest.fixture
def tiny_encoder(tmp_path):
    path = tmp_path / 'encoder'
    config = Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=24, conv_dim=[4]*7, conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2], num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8, apply_spec_augment=False, feat_extract_norm='layer',
        hidden_dropout=0., attention_dropout=0., feat_proj_dropout=0., layerdrop=0., activation_dropout=0.)
    Wav2Vec2Model(config).save_pretrained(path)
    return path


def model_config(path, cost='jsd', weight=.1):
    return wvOTTCConfig(wav2vec2_path=str(path), vocab_size=5, bins_metric=cost,
        forward_type='regular', wav2vec2_output_dropout=0.,
        temporal_smoothing_weight=weight, temporal_smoothing_tau=4.)


def audio_batch():
    torch.manual_seed(8)
    values = torch.randn(2, 1920)
    mask = torch.arange(1920)[None] < torch.tensor([1920, 1600])[:, None]
    values[~mask] = 0.
    return dict(input_values=values, attention_mask=mask.long(), labels=torch.tensor([[1, 2, 3], [2, 3, -100]]))


@pytest.mark.parametrize('cost', ['crossentropy', 'jsd', 'jsd_standard'])
def test_actual_ottc_loss_gradient_inference_and_disabled_replay(tiny_encoder, cost):
    base = DeviceOTTC(model_config(tiny_encoder, cost)).eval()
    new = TinySmoothedOTTC(model_config(tiny_encoder, cost)).eval()
    new.load_state_dict(base.state_dict(), strict=True)
    batch = audio_batch()
    with cpu_loss_allocations():
        before, after = base(**batch), new(**batch)
        torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
        penalty = smooth(after.logits, after.hidden_states[1], reduction='sum')
        torch.testing.assert_close(after.loss, before.loss + .1 * penalty)
        assert penalty.item() > 0
        bg = torch.autograd.grad(before.loss, (base.lm_head.weight, base.lm_weight.weight))
        ng = torch.autograd.grad(after.loss, (new.lm_head.weight, new.lm_weight.weight))
        assert not torch.equal(bg[0], ng[0])
        torch.testing.assert_close(bg[1], ng[1], rtol=0, atol=0)  # No direct alpha regularization.
        new.config.temporal_smoothing_weight = 0.
        zero = new(**batch)
        torch.testing.assert_close(zero.loss, before.loss, rtol=0, atol=0)
        new.config.temporal_smoothing_weight = .1
        tup = new(**batch, return_dict=False)
        assert len(tup) == 2
        torch.testing.assert_close(tup[0], after.loss)
    # No transport call or labels in audio-only inference, for either dict mode.
    audio = {k: v for k, v in batch.items() if k != 'labels'}
    with patch.object(upstream, 'batched_ottc_loss_bucketized', side_effect=AssertionError('inference transport')):
        a, b = base(**audio), new(**audio)
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        assert list(a.keys()) == list(b.keys()) and b.loss is None
        assert len(new(**audio, return_dict=False)) == 1


def test_save_reload_preserves_weight_tau_and_state(tiny_encoder, tmp_path):
    model = TinySmoothedOTTC(model_config(tiny_encoder)).eval()
    target = tmp_path / 'saved'
    model.save_pretrained(target)
    reloaded, loading = TinySmoothedOTTC.from_pretrained(target, output_loading_info=True)
    assert all(not loading[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
    assert reloaded.config.temporal_smoothing_weight == .1
    assert reloaded.config.temporal_smoothing_tau == 4.
    assert reloaded.config.temporal_smoothing_reduction == 'utterance_mean_then_batch_sum'
    assert reloaded.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)


def test_trainer_step_and_local_component_logs(tiny_encoder, tmp_path):
    model = TinySmoothedOTTC(model_config(tiny_encoder))
    rows = [{k: v[i] for k, v in audio_batch().items()} for i in range(2)]
    arguments = TrainingArguments(output_dir=str(tmp_path / 'trainer'), max_steps=1,
        per_device_train_batch_size=2, learning_rate=1e-3, logging_steps=1, save_strategy='no',
        report_to=[], use_cpu=True, disable_tqdm=True, dataloader_pin_memory=False)
    model.wav2vec2.gradient_checkpointing_disable()
    initial = model.lm_head.weight.detach().clone()
    trainer = TemporalSmoothingTrainer(model=model, args=arguments, train_dataset=rows)
    with cpu_loss_allocations():
        trainer.train()
    assert not torch.equal(initial, model.lm_head.weight)
    logs = next(r for r in trainer.state.log_history if 'loss_ottc_per_utterance' in r)
    assert logs['loss_temporal_per_utterance'] > 0
    assert abs(logs['loss_total_per_utterance'] - logs['loss_ottc_per_utterance'] -
               logs['loss_temporal_weighted_per_utterance']) < 1e-5
    trainer.save_state()
    saved = json.loads((tmp_path / 'trainer/trainer_state.json').read_text())
    assert any('loss_temporal_per_utterance' in r for r in saved['log_history'])


@pytest.mark.parametrize('head', ['mono', 's3t2ss'])
@pytest.mark.parametrize('cost', ['ce', 'jsd_upstream', 'jsd_standard'])
def test_native10_factory_preserves_trainability_and_cost(tiny_encoder, head, cost):
    from reproduction.ottc_temporal_smoothing_20261021.integration import recipe, build_model
    config = recipe(f'wav2vec2__timit__10ms__{head}__{cost}', .2)
    vocab = {'<pad>': 0, 'aa': 1, 'b': 2, 'sil': 3, '<unk>': 4}
    loader = Wav2Vec2Model.from_pretrained
    with patch.object(upstream.Wav2Vec2Model, 'from_pretrained', side_effect=lambda *a, **k: loader(tiny_encoder)):
        model, topo = build_model(config, vocab, 2)
    assert model.jsd == {'ce': False, 'jsd_upstream': True, 'jsd_standard': 'standard'}[cost]
    assert model.config.temporal_smoothing_weight == .2
    cnn = model.wav2vec2.feature_extractor.conv_layers
    assert cnn[-1].conv.stride == (1,)
    assert all(not p.requires_grad for layer in cnn[:-1] for p in layer.parameters())
    assert all(p.requires_grad for p in cnn[-1].parameters())
    batch = audio_batch()
    if topo:
        expanded = [torch.tensor([s for t in row.tolist() if t >= 0 for s in topo['mapping'][t]]) for row in batch['labels']]
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(expanded, batch_first=True, padding_value=-100)
    model.train()
    with cpu_loss_allocations():
        out = model(**batch)
    assert out.logits.shape[1] == 10
    assert out.hidden_states[1].shape == out.logits.shape[:2]
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for parameters in [cnn[-1].parameters(), model.lm_head.parameters(), model.lm_weight.parameters()]:
        parameters = list(parameters)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert sum(float(p.grad.square().sum()) for p in parameters) > 0
