from dataclasses import asdict
import json
import pytest
import torch
from reproduction.ottc_temporal_smoothing_20261021.test_smoothing import (
    tiny_encoder, cpu_loss_allocations, DeviceOTTC,
)
from ottc.models.sequence.wav2vec2_ottc import wvOTTCConfig
from ottc.models.sequence.wav2vec2_ottc_gw_outputfix import GWTransportMixin
from ottc.ops_loss.gw_transport import GWOptions
from reproduction.timit_gw_outputfix_20261027.diagnostics import summarize, edit_counts


class TinyGW(GWTransportMixin, DeviceOTTC):
    pass


@pytest.mark.parametrize('mode,weight', [('monotone', 0.), ('monotone', .1), ('asot', .1)])
def test_actual_model_backward_and_audio_only(tiny_encoder, mode, weight):
    torch.manual_seed(4)
    config = wvOTTCConfig(wav2vec2_path=str(tiny_encoder), vocab_size=5, bins_metric='ag',
        forward_type='regular', wav2vec2_output_dropout=0.,
        gw_options=asdict(GWOptions(mode=mode, weight=weight, iterations=3)))
    model = TinyGW(config).eval()
    samples = torch.randn(2, 2400)
    mask = torch.ones(2, 2400, dtype=torch.long); mask[1, 1700:] = 0
    labels = torch.tensor([[1, 2, 1], [2, 3, -100]])
    with cpu_loss_allocations():
        out = model(samples, attention_mask=mask, labels=labels)
        from accelerate.utils.operations import convert_to_fp32
        converted = convert_to_fp32(out)
        assert isinstance(converted.loss, torch.Tensor)
        assert 'loss' in converted and converted['loss'] is converted.loss
        torch.testing.assert_close(converted.loss, out.loss)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        for head in (model.lm_head, model.lm_weight):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            assert sum(float(p.grad.square().sum()) for p in head.parameters()) > 0
        audio = model(samples, attention_mask=mask)
        torch.testing.assert_close(out.logits, audio.logits)
        assert audio.attentions is None and audio.loss is None
        # Label substitution never changes the audio encoder/classifier output.
        other = model(samples, attention_mask=mask, labels=labels.flip(0))
        torch.testing.assert_close(out.logits, other.logits)
        if mode == 'monotone' and weight == 0:
            base = DeviceOTTC(config).eval(); base.load_state_dict(model.state_dict())
            expected = base(samples, attention_mask=mask, labels=labels)
            assert torch.equal(out.loss, expected.loss)


def test_aba_and_short_reference_deletion_diagnostic(tmp_path):
    path = tmp_path/'predictions.jsonl'
    path.write_text(json.dumps(dict(utt='one', reference=[['aa',0,.05],['ae',.05,.06]],
        text_free=dict(hypothesis=['aa'], frame_token_ids=[1,1,2,1,1])))+'\n')
    result = summarize(path, {'aa':1,'ae':2})['counts']
    assert result['aba_middle_le20ms'] == 1
    assert result['D'] == 1 and result['short_reference_deleted'] == 1


def test_edit_counts():
    counts, deleted = edit_counts(['aa','ae','iy'], ['aa','eh','iy','aa'])
    assert counts == dict(S=1,I=1,D=0)
    assert deleted == []


def test_epoch_retention_and_fp32_average(tmp_path):
    from types import SimpleNamespace
    from reproduction.timit_gw_outputfix_20261027.train import Policy
    from reproduction.timit_gw_outputfix_20261027.evaluate import average
    model = torch.nn.Linear(2, 1)
    model.config = SimpleNamespace(to_dict=lambda: {'kind': 'test'})
    config = dict(name='test', source_sha256={})
    policy = Policy(tmp_path, config, [], {}, None, None, float('inf'), None)
    for epoch in range(1, 12):
        with torch.no_grad():
            model.weight.fill_(epoch); model.bias.fill_(epoch)
        state = SimpleNamespace(epoch=float(epoch), global_step=epoch*3)
        control = SimpleNamespace(should_save=False)
        policy.on_epoch_end(None, state, control, model=model)
        assert control.should_save
    paths = sorted((tmp_path/'epoch_weights').glob('*.pt'))
    assert len(paths) == 11
    output = tmp_path/'average.pt'
    average(paths[-10:], output, config)
    saved = torch.load(output, weights_only=False)
    assert len(saved['members']) == 10
    for value in saved['model'].values():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, torch.full_like(value, 6.5))


@pytest.mark.parametrize('weight', [0., .1])
def test_actual_trainer_bf16_optimizer_steps(tiny_encoder, tmp_path, weight):
    from transformers import TrainingArguments
    from reproduction.timit_gw_outputfix_20261027.train import QuietTrainer
    config = wvOTTCConfig(wav2vec2_path=str(tiny_encoder), vocab_size=5,
        bins_metric='ag', forward_type='regular', wav2vec2_output_dropout=0.,
        gw_options=asdict(GWOptions(mode='asot', weight=weight, iterations=3)))
    model = TinyGW(config)
    before = model.lm_head.weight.detach().clone()
    items = [dict(input_values=torch.randn(2400), attention_mask=torch.ones(2400,dtype=torch.long),
                  labels=torch.tensor([1,2,1])) for _ in range(4)]
    def collate(rows):
        return dict(microbatches=[{k:torch.stack([r[k] for r in rows]) for k in rows[0]}],
                    logical_batch_size=len(rows))
    args = TrainingArguments(output_dir=str(tmp_path/'trainer'), use_cpu=True, bf16=True,
        per_device_train_batch_size=2, max_steps=2, save_strategy='no', report_to=[],
        remove_unused_columns=False, disable_tqdm=True)
    trainer = QuietTrainer(model=model, args=args, train_dataset=items, data_collator=collate)
    with cpu_loss_allocations():
        result = trainer.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert not torch.equal(model.lm_head.weight, before)
