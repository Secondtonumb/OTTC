"""CPU objective, PHN coverage, gradient, FP32 averaging and exact-resume checks."""
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import TrainingArguments, TrainerCallback, set_seed
from reproduction.timit_frame_ce_20261026 import common
from reproduction.timit_frame_ce_20261026.model import frame_loss
from reproduction.timit_frame_ce_20261026.profile_recipe import gradient_checks
from reproduction.timit_ssl_native10_20261019.validate import TinyLoader
import ottc.models.sequence.wav2vec2_ottc as upstream


def tiny(config, vocab):
    original = upstream.Wav2Vec2Model
    upstream.Wav2Vec2Model = TinyLoader
    try:
        model, _ = common.build_model(config, vocab, 2)
    finally:
        upstream.Wav2Vec2Model = original
    model.dropout.p = 0.
    return model


class Data:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return dict(input_values=torch.sin(torch.arange(5120).float() * (.01 + index * .001)),
            attention_mask=torch.ones(5120, dtype=torch.long), labels=torch.tensor([1, 2, 3]),
            frame_targets=torch.tensor([1]*10 + [2]*10 + [3]*10))


def collate(rows):
    return dict(microbatches=[{k: torch.stack([r[k] for r in rows]) for k in rows[0]}], logical_batch_size=len(rows))


class Stop(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
            control.should_save = True


def resume_case(path, config, vocab, resume=False, stop=False):
    from reproduction.timit_frame_ce_20261026.train import QuietTrainer, Policy
    class Run:
        def log(self, *args, **kwargs):
            pass
    set_seed(42)
    model = tiny(config, vocab)
    path.mkdir(exist_ok=True)
    policy = Policy(path, config, [], vocab, None, None, float('inf'), Run())
    args = TrainingArguments(output_dir=str(path), per_device_train_batch_size=2, num_train_epochs=2,
        learning_rate=1e-4, warmup_steps=1, save_strategy='no', save_total_limit=1, logging_steps=1,
        report_to=[], dataloader_num_workers=0, remove_unused_columns=False, disable_tqdm=True,
        seed=42, save_safetensors=True, use_cpu=True)
    trainer = QuietTrainer(model=model, args=args, train_dataset=Data(), data_collator=collate,
                           callbacks=[policy] + ([Stop()] if stop else []))
    trainer.train(resume_from_checkpoint=str(path / 'checkpoint-1') if resume else None)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    torch.set_num_threads(2)
    config = common.recipe('frame_ce__seed42')
    data, rows, vocab, processor, audit = common.load_data(config)
    assert audit == __import__('json').loads((Path(config['baseline']) / 'run_config.json').read_text())['data']
    # Independent analytic target: RF centers12.46875,22.46875,...ms.
    target = common.project([['a', 0., .02], ['b', .02, .1]], {'a': 1, 'b': 2}, 1600)
    assert target.tolist() == [1] + [2]*7
    partial = common.project([['b', .02, .1]], {'b': 2}, 1600)
    assert partial.tolist() == [-100] + [2]*7
    for bad in ([['a', 0., .03], ['b', .02, .1]], [['missing', 0., .1]]):
        try:
            common.project(bad, {'a': 1, 'b': 2}, 1600)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('Invalid PHN accepted')
    logits = torch.tensor([[[2., 0., -1.], [0., 2., -1.]], [[1., 2., 0.], [50., -50., 0.]]], requires_grad=True)
    targets = torch.tensor([[0, 1], [1, -100]])
    mask = targets.ne(-100).float()
    expected = F.cross_entropy(logits[0], targets[0]) + F.cross_entropy(logits[1, :1], targets[1, :1])
    actual = frame_loss(logits, targets, mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.equal(logits.grad[1, 1], torch.zeros(3))
    set_seed(42)
    model = tiny(config, vocab)
    batch = collate([Data()[0], Data()[1]])['microbatches'][0]
    before = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.detach().clone()
    result = model(**batch)
    assert result.logits.shape == (2, 30, 42) and torch.isfinite(result.loss)
    result.loss.backward()
    gradients = gradient_checks(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    assert torch.equal(before, model.wav2vec2.feature_extractor.conv_layers[0].conv.weight)
    model.eval()
    with torch.inference_mode():
        first = model(input_values=batch['input_values'], attention_mask=batch['attention_mask']).logits
        second = model(input_values=batch['input_values'], attention_mask=batch['attention_mask'],
                       frame_targets=torch.zeros_like(batch['frame_targets'])).logits
    assert torch.equal(first, second)
    real = common.Collator(processor)([data['train'][0], data['train'][1]])
    for i, row in enumerate((data['train'][0], data['train'][1])):
        import soundfile as sf
        assert int(real['frame_targets'][i].ne(-100).sum()) == common.frame_count(sf.info(row['wav']).frames)
    with tempfile.TemporaryDirectory(prefix='timit_frame_ce_') as temporary:
        path = Path(temporary)
        common.atomic_weights(model, path / 'first.pt', dict(recipe=config['name'], epoch=1, step=1))
        common.atomic_weights(model, path / 'second.pt', dict(recipe=config['name'], epoch=2, step=2))
        from reproduction.timit_frame_ce_20261026.evaluate import average
        average([path / 'first.pt', path / 'second.pt'], path / 'avg.pt', config)
        saved = torch.load(path / 'avg.pt', weights_only=False)
        assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(saved['model'][key], value.cpu())
        full = resume_case(path / 'continuous', config, vocab)
        resume_case(path / 'resumed', config, vocab, stop=True)
        resumed = resume_case(path / 'resumed', config, vocab, resume=True)
        delta = max(float((full[k]-resumed[k]).abs().max()) for k in full)
        assert delta == 0., delta
        assert (path / 'continuous/epoch_weights/epoch-001.pt').exists()
        assert (path / 'continuous/epoch_weights/epoch-002.pt').exists()
    common.write(common.HERE / 'CPU_PASS.json', dict(source_sha256=common.source_hashes(), data=audit,
        all_5212_timelines_validated=True, coverage=processor.frame_ce_coverage, loss_masking_and_reduction=True, gradient_groups=gradients,
        audio_only_inference=True, fp32_saved_and_averaged=True, exact_resume_max_difference=delta,
        independent_epoch_weights=True, real_audio_padding_verified=True))
    print('CPU_PASS')


if __name__ == '__main__':
    main()
