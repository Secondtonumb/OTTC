"""Local gates: real tiny OTTC, five-epoch schedule, final-three mean and resume."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import TrainerCallback, TrainingArguments, Wav2Vec2Config, Wav2Vec2Model

from reproduction.timit_divergences_20261022 import common
from reproduction.timit_divergences_20261022.evaluate import average_checkpoints
from reproduction.timit_divergences_20261022.train import Policy, QuietTrainer
from ottc.models.sequence import wav2vec2_ottc as upstream


@pytest.mark.parametrize('cost', common.COSTS)
def test_real_model_forward_backward_and_native_stride(cost):
    encoder = Wav2Vec2Model(Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
        conv_dim=[4]*7, conv_stride=[5,2,2,2,2,2,2], conv_kernel=[10,3,3,3,3,2,2],
        num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=8, apply_spec_augment=False,
        hidden_dropout=0., attention_dropout=0., feat_proj_dropout=0., layerdrop=0.,
        activation_dropout=0., feat_extract_norm='layer'))
    config = common.recipe(cost)
    with patch.object(upstream.Wav2Vec2Model, 'from_pretrained', return_value=encoder):
        model, topo = common.build_model(config, {}, 73)
    assert topo is None
    assert model.config.bins_metric == cost
    assert model.wav2vec2.feature_extractor.conv_layers[-1].conv.stride == (1,)
    original = upstream.batched_ottc_loss_bucketized
    source = inspect.getsource(original).replace(".to('cuda')", ".to('cpu')")
    namespace = dict(original.__globals__)
    exec(compile(source, '<CPU allocations only>', 'exec'), namespace)
    with patch.object(upstream, original.__name__, namespace[original.__name__]):
        out = model(input_values=torch.randn(2, 1920), attention_mask=torch.ones(2, 1920,dtype=torch.long),
                    labels=torch.tensor([[1,2,3], [2,3,-100]]))
        assert torch.isfinite(out.loss)
        out.loss.backward()
    for module in (model.lm_head, model.lm_weight, encoder.feature_extractor.conv_layers[-1]):
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(not p.requires_grad and p.grad is None for layer in encoder.feature_extractor.conv_layers[:-1] for p in layer.parameters())


def make_policy(output, config):
    run = SimpleNamespace(log=lambda _: None)
    return Policy(output, config, [], {}, None, None, float('inf'), run)


def test_requested_checkpoint_schedule(tmp_path):
    config = common.recipe('jdiv')
    policy = make_policy(tmp_path, config)
    model = torch.nn.Linear(2, 1)
    model.config = SimpleNamespace(to_dict=lambda: {'bins_metric':'jdiv'})
    evaluated = []
    policy.on_evaluate = lambda args, state, control, **kw: evaluated.append(state.epoch)
    full_saves = []
    for epoch in range(1, 41):
        state = SimpleNamespace(epoch=float(epoch), global_step=epoch*73)
        control = SimpleNamespace(should_save=False)
        policy.on_epoch_end(None, state, control, model=model)
        if control.should_save:
            full_saves.append(epoch)
    assert evaluated == full_saves == [5,10,15,20,25,30,35,40]
    weights = sorted(int(p.stem.split('-')[1]) for p in (tmp_path/'epoch_weights').glob('*.pt'))
    assert weights == [5,10,15,20,25,30,35,38,39,40]
    partial = SimpleNamespace(epoch=38.5, global_step=2810)
    policy.on_epoch_end(None, partial, SimpleNamespace(should_save=False), model=model)
    assert evaluated == full_saves
    policy.stop_requested = True
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    policy.on_step_end(None, partial, control)
    assert control.should_save and control.should_training_stop


def write_member(path, epoch, value, identity='same'):
    torch.save(dict(model={'w':torch.tensor([value, value+1.]), 'counter':torch.tensor(7)},
        epoch=epoch, step=epoch*73, recipe=identity, config_sha256='cfg', source_sha256={}), path)


def test_final_three_arithmetic_mean_and_provenance(tmp_path):
    paths = [tmp_path/f'{epoch}.pt' for epoch in (38,39,40)]
    for path, epoch in zip(paths, (38,39,40)):
        write_member(path, epoch, float(epoch))
    target = average_checkpoints(paths, tmp_path/'average.pt')
    saved = torch.load(target, weights_only=False)
    torch.testing.assert_close(saved['model']['w'], torch.tensor([39.,40.]))
    assert saved['model']['counter'].item() == 7
    manifest = json.loads(target.with_suffix('.json').read_text())
    assert [m['epoch'] for m in manifest['members']] == [38,39,40]
    assert all(common.sha(Path(m['path'])) == m['sha256'] for m in manifest['members'])
    write_member(paths[-1], 40, 40., identity='different')
    with pytest.raises(ValueError, match='identities'):
        average_checkpoints(paths, tmp_path/'bad.pt')


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 1)
        self.config = SimpleNamespace(to_dict=lambda: {'model_type':'local-test'})
    def forward(self, input_values, labels):
        return SimpleNamespace(loss=(self.head(input_values).flatten()-labels).square().sum())


def toy_collator(rows):
    batch = {k:torch.stack([r[k] for r in rows]) for k in rows[0]}
    return dict(microbatches=[batch], logical_batch_size=len(rows))


class StopAfterThree(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 3:
            control.should_save = control.should_training_stop = True


def test_trainer_full_resume_after_partial_epoch(tmp_path):
    # Actual HF optimizer/scheduler/RNG save and resume; tiny linear model only.
    data = [dict(input_values=torch.tensor([i/10., 1.]), labels=torch.tensor(i/5.)) for i in range(4)]
    args = TrainingArguments(output_dir=str(tmp_path), per_device_train_batch_size=2,
        num_train_epochs=6, evaluation_strategy='no', save_strategy='no', save_total_limit=1,
        logging_steps=1, report_to=[], disable_tqdm=True, use_cpu=True,
        remove_unused_columns=False, seed=42, learning_rate=.01)
    config = common.recipe('kl', epochs=6, batch_size=2)
    policy = make_policy(tmp_path, config)
    evaluated = []
    policy.on_evaluate = lambda args, state, control, **kw: evaluated.append(state.epoch)
    torch.manual_seed(7)
    initial = ToyModel().state_dict()
    model = ToyModel(); model.load_state_dict(initial)
    trainer = QuietTrainer(model=model, args=args, train_dataset=data, data_collator=toy_collator,
                           callbacks=[policy, StopAfterThree()])
    trainer.train()
    checkpoint = tmp_path/'checkpoint-3'
    assert all((checkpoint/f).exists() for f in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'))
    assert not list((tmp_path/'epoch_weights').glob('*.pt')) if (tmp_path/'epoch_weights').exists() else True
    resumed_model = ToyModel()
    resumed = QuietTrainer(model=resumed_model, args=args, train_dataset=data, data_collator=toy_collator,
                           callbacks=[policy])
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 12 and resumed.state.epoch == 6
    assert evaluated == [5., 6.]
    assert (tmp_path/'epoch_weights/epoch-004.pt').exists()
    assert (tmp_path/'epoch_weights/epoch-005.pt').exists()
    assert (tmp_path/'epoch_weights/epoch-006.pt').exists()
    assert len(list(tmp_path.glob('checkpoint-*'))) == 1
    # Compare final parameters with uninterrupted training at identical seed.
    full_dir = tmp_path/'uninterrupted'
    args2 = TrainingArguments(output_dir=str(full_dir), per_device_train_batch_size=2,
        num_train_epochs=6, evaluation_strategy='no', save_strategy='no', logging_steps=1,
        report_to=[], disable_tqdm=True, use_cpu=True, remove_unused_columns=False,
        seed=42, learning_rate=.01)
    full_model = ToyModel(); full_model.load_state_dict(initial)
    QuietTrainer(model=full_model, args=args2, train_dataset=data, data_collator=toy_collator).train()
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[key], value, atol=0, rtol=0)
