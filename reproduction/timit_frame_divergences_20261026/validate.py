"""Validate exact costs, masked reduction, all model gradients and exact resume."""
import json
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import TrainingArguments, set_seed
from reproduction.timit_frame_divergences_20261026 import common
from reproduction.timit_frame_divergences_20261026.model import frame_loss, per_frame
from reproduction.timit_frame_divergences_20261026.profile_recipe import gradient_checks
from reproduction.timit_frame_ce_20261026.validate import Data, collate, Stop, TinyLoader
import ottc.models.sequence.wav2vec2_ottc as upstream
from ottc.ops_loss import ot_loss


def tiny(config, vocab):
    original = upstream.Wav2Vec2Model
    upstream.Wav2Vec2Model = TinyLoader
    try:
        model, _ = common.build_model(config, vocab, 2)
    finally:
        upstream.Wav2Vec2Model = original
    model.dropout.p = 0.
    return model


def independent(logits, targets, cost):
    q = F.one_hot(targets, logits.shape[-1]).float()
    p = logits.softmax(-1)
    if cost == 'jsd_standard':
        m = (p + q) / 2
        return .5 * (F.kl_div(m.log(), p, reduction='none').sum(-1)
                     + F.kl_div(m.log(), q, reduction='none').sum(-1))
    if cost == 'jdiv':
        qe = q.clamp_min(1e-12)
        qe = qe / qe.sum(-1, keepdim=True)
        return (p*(p.log()-qe.log()) + qe*(qe.log()-p.log())).sum(-1)
    qe = q + (1-q)*1e-12 - q*1e-12
    m = (p + qe) / 2
    return .5 * ((m*(m.log()-p.log())).sum(-1) + (m*(m.log()-qe.log())).sum(-1))


def resume_case(path, config, vocab, resume=False, stop=False):
    from reproduction.timit_frame_divergences_20261026.train import QuietTrainer, Policy
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
    data, rows, vocab, processor, audit = common.load_data(common.recipe('frame_ag__seed42'))
    assert audit == json.loads((Path(common.recipe('frame_ag__seed42')['baseline']) / 'run_config.json').read_text())['data']
    checks = {}
    for cost in common.COSTS:
        config = common.recipe(f'frame_{cost}__seed42')
        logits = torch.tensor([[[2., 0., -1.], [0., 2., -1.]], [[1., 2., 0.], [1000., -1000., 0.]]], requires_grad=True)
        targets = torch.tensor([[0, 1], [1, -100]])
        mask = torch.ones_like(targets, dtype=torch.float32)
        actual = frame_loss(logits, targets, mask, cost)
        expected = independent(logits[0], targets[0], cost).mean() + independent(logits[1, :1], targets[1, :1], cost).mean()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)
        assert torch.equal(actual_grad[1, 1], torch.zeros(3))
        mixed = per_frame(logits[0].detach().bfloat16(), targets[0], cost)
        assert mixed.dtype == torch.float32 and torch.isfinite(mixed).all()
        if cost == 'ag':
            q = F.one_hot(targets[0], 3).float()
            assert torch.equal(per_frame(logits[0], targets[0], cost), ot_loss.jensen_shannon_divergence(logits[0].softmax(-1), q).sum(-1))
            try:
                per_frame(logits[1, 1:], torch.tensor([1]), cost)
            except FloatingPointError:
                pass
            else:
                raise AssertionError('AG underflow silently accepted')
        else:
            extreme = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
            value = per_frame(extreme, torch.tensor([1]), cost)
            gradient, = torch.autograd.grad(value.sum(), extreme)
            assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
        set_seed(42)
        model = tiny(config, vocab)
        batch = collate([Data()[0], Data()[1]])['microbatches'][0]
        before = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight.detach().clone()
        out = model(**batch)
        assert out.logits.shape == (2, 30, 42) and torch.isfinite(out.loss)
        out.loss.backward()
        gradients = gradient_checks(model)
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
        assert torch.equal(before, model.wav2vec2.feature_extractor.conv_layers[0].conv.weight)
        model.eval()
        with torch.inference_mode():
            a = model(input_values=batch['input_values'], attention_mask=batch['attention_mask']).logits
            b = model(input_values=batch['input_values'], attention_mask=batch['attention_mask'],
                      frame_targets=torch.zeros_like(batch['frame_targets'])).logits
        assert torch.equal(a, b)
        with tempfile.TemporaryDirectory(prefix=f'timit_frame_{cost}_') as temporary:
            path = Path(temporary)
            for epoch in (1, 2):
                common.atomic_weights(model, path / f'{epoch}.pt', dict(recipe=config['name'], epoch=epoch, step=epoch))
            from reproduction.timit_frame_divergences_20261026.evaluate import average
            average([path / '1.pt', path / '2.pt'], path / 'avg.pt', config)
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
        checks[cost] = dict(loss=float(actual), analytic_loss_and_gradient_match=True,
            padding_and_uncovered_ignored=True, bf16_cost_promoted_to_fp32=True, gradient_groups=gradients,
            audio_only_inference=True, fp32_saved_and_averaged=True, exact_resume_max_difference=delta,
            legacy_AG_bitwise_equal=cost == 'ag')
    common.write(common.HERE / 'CPU_PASS.json', dict(source_sha256=common.source_hashes(), data=audit,
        coverage=processor.frame_ce_coverage, checks=checks, all_5212_timelines_validated=True))
    print('CPU_PASS', list(checks))


if __name__ == '__main__':
    main()
