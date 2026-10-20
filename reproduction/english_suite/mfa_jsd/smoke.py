"""GPU validation of the existing OTTC jsd loss wired into Zapformer."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
sys.path.insert(0, str(SUITE))
import torch
import torch.nn.functional as F
from mfa_jsd.loss import SOURCE, SOURCE_SHA256, original_ottc, acoustic_columns, batched_ottc_jsd_loss
from mfa_jsd.model_jsd import MFAZapformer
from model_10ms import MFAZapformer as CEZapformer
from ifmdd_losses import make_ottc_inputs


def loss_checks():
    torch.manual_seed(43)
    logits = torch.randn(2, 7, 105, device='cuda', requires_grad=True)
    weights = torch.randn(2, 7, device='cuda', requires_grad=True)
    lengths = torch.tensor([7, 4], device='cuda')
    labels = torch.tensor([[3, 4, 5], [4, 6, 0]], device='cuda')
    target_lengths = torch.tensor([3, 2], device='cuda')
    fw, tw, fm, onehot = make_ottc_inputs(logits, lengths, labels, target_lengths, weights)
    tm = torch.arange(3, device='cuda')[None] < target_lengths[:, None]
    value = batched_ottc_jsd_loss(logits, onehot, fw, tw, fm, tm)
    selected, targets = acoustic_columns(logits, onehot)
    direct = original_ottc.batched_ottc_loss_bucketized(selected, targets, fw, tw, amask=fm, bmask=tm, jsd=True)[0]
    torch.testing.assert_close(value, direct, rtol=0, atol=0)
    ga = torch.autograd.grad(value, (logits, weights), retain_graph=True)
    gb = torch.autograd.grad(direct, (logits, weights), retain_graph=True)
    for a, b in zip(ga, gb):torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0) for g in ga)
    assert not bool(ga[0][..., 1:3].any())
    assert not bool(ga[0][1, 4:].any()) and not bool(ga[1][1, 4:].any())
    # Independent dense interval-overlap oracle; use the ORIGINAL pair cost.
    oracle = value * 0
    for b in range(2):
        af = fw[b, :lengths[b]].cumsum(0)
        at = tw[b, :target_lengths[b]].cumsum(0)
        starts_f = F.pad(af[:-1], (1, 0))
        starts_t = F.pad(at[:-1], (1, 0))
        overlaps = (torch.minimum(af[:, None], at[None]) - torch.maximum(starts_f[:, None], starts_t[None])).clamp_min(0)
        p = selected[b, :lengths[b]].softmax(-1)
        q = targets[b, :target_lengths[b]].float()
        costs = original_ottc.jensen_shannon_divergence(
            p[:, None].expand(-1, len(q), -1).reshape(-1, 103),
            q[None].expand(len(p), -1, -1).reshape(-1, 103)
        ).sum(-1).reshape(len(p), len(q))
        oracle = oracle + (overlaps * costs).sum()
    torch.testing.assert_close(value, oracle, rtol=2e-6, atol=2e-6)
    # Reproduce the actual control-mask failure before the phone-column adapter.
    masked = logits.detach().index_fill(-1, torch.tensor([1, 2], device='cuda'), -1e4)
    broken = original_ottc.batched_ottc_loss_bucketized(masked, onehot, fw.detach(), tw, amask=fm, bmask=tm, jsd=True)[0]
    assert not bool(torch.isfinite(broken))
    return {'adapter_equals_original_loss_and_gradients': True, 'dense_transport_oracle': True,
            'padding_and_control_gradients_zero': True, 'unadapted_control_mask_nonfinite': True,
            'loss': float(value.detach())}


def model_check(tiny):
    torch.manual_seed(42)
    model = MFAZapformer(tiny=tiny).cuda().train()
    # The existing CE class remains bound to its original CE loss.
    assert CEZapformer.forward.__globals__['batched_ottc_loss_bucketized'] is not batched_ottc_jsd_loss
    features = torch.randn(2, 64, 80, device='cuda')
    lengths = torch.tensor([64, 53], device='cuda')
    targets = torch.tensor([3, 7, 5, 4, 6], device='cuda')
    target_lengths = torch.tensor([3, 2], device='cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, logits, out_lengths = model(features, lengths, targets, target_lengths)
    assert bool(torch.isfinite(loss))
    assert logits.shape == (2, 64, 105) and torch.equal(lengths, out_lengths)
    loss.backward()
    groups = {'encoder': model.acoustic.encoder.parameters(),
              'projection': model.acoustic.encoder_embed.parameters(),
              'classifier': model.acoustic.ctc_output.parameters(),
              'duration_weights': model.weight_head.parameters()}
    gradients = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads), name
        gradients[name] = sum(float(g.abs().sum()) for g in grads)
        assert gradients[name] > 0, name
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
    optimizer.step()
    return {'tiny': tiny, 'loss': float(loss.detach()), 'grad_norm': norm,
            'gradients': gradients, 'output_grid_ms': 10, 'classes': 105,
            'parameters': sum(p.numel() for p in model.parameters()), 'optimizer_step': True}


def main():
    from reproduction.wandb_policy import init_evaluation
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    result = {'status': 'PASS', 'validation_only': True, 'production_training_started': False,
              'loss_source': str(SOURCE), 'loss_source_sha256': SOURCE_SHA256,
              'adapter_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in HERE.glob('*.py')},
              'gpu': torch.cuda.get_device_name(), 'loss_checks': loss_checks(),
              'models': [model_check(True), model_check(False)]}
    # Register smoke outcomes separately from formal experimental results.
    import wandb
    arguments = dict(project='ottc-mfa-english', entity='kevingenghaopeng-',
                     name='zapformer-upstream-jsd-smoke', job_type='validation',
                     config={'validation_only': True, 'objective': 'jsd_upstream',
                             'loss_source_sha256': SOURCE_SHA256}, dir=str(args.output),
                     mode=os.environ.get('WANDB_MODE', 'online'))
    try:run = init_evaluation(**arguments)
    except wandb.errors.CommError:
        arguments['mode'] = 'offline';run = init_evaluation(**arguments)
    run.summary.update(result)
    result['wandb'] = {'id': run.id, 'url': run.url}
    (args.output/'SMOKE_PASS.json').write_text(json.dumps(result, indent=2)+'\n')
    run.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':main()
