"""Offline, real tiny encoder; CPU adapts only native OT's CUDA allocations."""
import ast
import inspect

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from ottc.models.sequence import wav2vec2_ottc_film as film
from ottc.ops_loss import ot_loss


@pytest.fixture
def model(tmp_path, monkeypatch):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    encoder = Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, conv_dim=(8,), conv_stride=(2,), conv_kernel=(3,),
        num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=8,
        hidden_dropout=0., attention_dropout=0., activation_dropout=0.,
        feat_proj_dropout=0., layerdrop=0., mask_time_prob=0.))
    encoder.save_pretrained(tmp_path / 'encoder')
    config = film.wvOTTCFiLMConfig(wav2vec2_path=str(tmp_path / 'encoder'),
        vocab_size=5, wav2vec2_output_dropout=0., ctc_loss_weight=.3)
    # No loss substitution: keep the production transport/divergence arithmetic.
    source = inspect.getsource(ot_loss.batched_ottc_loss_bucketized)
    source = source.replace(".to('cuda')", ".to('cpu')")
    namespace = dict(vars(ot_loss))
    exec(compile(source, '<native OT CPU allocations>', 'exec'), namespace)
    monkeypatch.setattr(film, 'batched_ottc_loss_bucketized',
                        namespace['batched_ottc_loss_bucketized'])
    return film.wvForOTTCFiLM(config)


def batch():
    audio = torch.randn(2, 41)
    mask = torch.ones(2, 41, dtype=torch.long)
    mask[1, 31:] = 0
    audio[1, 31:] = 0
    return dict(input_values=audio, attention_mask=mask,
                labels=torch.tensor([[1, 2, 2], [3, 4, -100]]))


def test_exact_upstream_film_identity_and_parity():
    # Extract only the original class; importing its full SpeechBrain model is
    # unnecessary. This also records the source implementation being ported.
    path = '/work/gm64/m64000/IF-MDD/models/phn_mono_ssl_model_v3_refactored.py'
    from pathlib import Path
    if not Path(path).exists():
        pytest.skip('Original IF-MDD source is not installed')
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'ContextFiLM')
    from typing import Optional
    ns = dict(nn=torch.nn, torch=torch, Optional=Optional)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
    new, original = film.ContextFiLM(8, 5), ns['ContextFiLM'](8, 5)
    x, condition = torch.randn(2, 4, 8), torch.randn(2, 4, 5).softmax(-1)
    torch.testing.assert_close(new(x, condition), x, atol=0, rtol=0)
    with torch.no_grad():
        new.net[-1].weight.normal_(std=.1)
        new.net[-1].bias.normal_(std=.1)
    original.load_state_dict(new.state_dict())
    torch.testing.assert_close(new(x, condition), original(x, condition), atol=0, rtol=0)


def test_native_loss_gradients_masks_and_shared_targets(model):
    data = batch()
    out = model(**data)
    assert out.logits.shape == (2, 20, 5)
    mass, mask, target_mass = out.hidden_states
    assert mask.sum(-1).tolist() == [20, 15]
    assert torch.count_nonzero(mass[~mask]) == 0
    torch.testing.assert_close(mass.sum(-1), torch.ones(2))
    torch.testing.assert_close(target_mass.sum(-1), torch.ones(2))
    raw = model.pop_film_loss_statistics()
    torch.testing.assert_close(out.loss.detach(), raw[0] + .3 * raw[1])
    assert raw[-1].item() == 2
    out.loss.backward()
    for module in [model.wav2vec2, model.ctc_head, model.lm_head,
                   model.lm_weight, model.context_film.net[-1]]:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize('detach', [True, False])
def test_condition_gradient_policy(model, detach):
    model.config.film_detach_condition = detach
    with torch.no_grad():
        model.context_film.net[-1].weight.normal_(std=.1)
    data = batch()
    logits = model(data['input_values'], data['attention_mask']).logits
    grad, = torch.autograd.grad(logits.square().sum(), model.ctc_head.weight,
                               allow_unused=True)
    if detach:
        assert grad is None
    else:
        assert grad is not None and grad.abs().sum() > 0


def test_audio_only_tuple_ctc_and_checkpoint_roundtrip(model, tmp_path, monkeypatch):
    model.eval()
    data = batch()
    def no_transport(*args, **kwargs):
        raise AssertionError('Inference called transport')
    monkeypatch.setattr(film, 'batched_ottc_loss_bucketized', no_transport)
    with torch.no_grad():
        out = model(data['input_values'], data['attention_mask'])
        ctc = model.ctc_forward(data['input_values'], data['attention_mask'])
        assert ctc.logits.shape == out.logits.shape
        assert len(model(data['input_values'], data['attention_mask'], return_dict=False)) == 1
        torch.testing.assert_close(out.logits, model.lm_head(model.dropout(
            model.wav2vec2(data['input_values'], attention_mask=data['attention_mask']).last_hidden_state)))
        model.save_pretrained(tmp_path / 'roundtrip')
        restored = film.wvForOTTCFiLM.from_pretrained(tmp_path / 'roundtrip').eval()
        torch.testing.assert_close(out.logits, restored(data['input_values'], data['attention_mask']).logits,
                                   atol=0, rtol=0)
    assert all(p.dtype == torch.float32 for p in restored.parameters())
    from safetensors.torch import load_file
    assert all(t.dtype == torch.float32 for t in load_file(str(tmp_path / 'roundtrip/model.safetensors')).values()
               if t.is_floating_point())
    assert restored.config.film_detach_condition is True
    assert model.pop_film_loss_statistics() is None


@pytest.mark.parametrize('labels, message', [
    ([[0, 1], [2, 3]], 'blank'), ([[1, -100, 2], [2, 3, 4]], 'right padded'),
    ([[-100, -100], [2, 3]], 'valid monophone'), ([[5, 1], [2, 3]], 'valid monophone'),
    ([[1] * 15, [2] * 15], 'Insufficient')])
def test_invalid_targets_fail_clearly(model, labels, message):
    data = batch()
    data['labels'] = torch.tensor(labels)
    with pytest.raises(ValueError, match=message):
        model(**data)


def test_config_guards_and_default_cost():
    assert film.wvOTTCFiLMConfig().bins_metric == 'jsd'
    for options in [dict(ctc_loss_weight=0), dict(ottc_loss_weight=float('nan')),
                    dict(forward_type='regfrozen'), dict(blank_id=100),
                    dict(film_dropout=1), dict(film_hidden_size=-1)]:
        with pytest.raises(ValueError):
            film.wvOTTCFiLMConfig(**options)
