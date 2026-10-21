"""Bounded real-CUDA loss checks and real TIMIT native10ms batch probe.

Local-only smoke test: never writes model checkpoints or initializes W&B.
"""
import argparse
import ast
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'reproduction/vendor'), str(ROOT)]

import torch
import torch.nn.functional as F
from ottc.ops_loss import ot_loss

LOG = logging.getLogger('ottc_smoothing_probe')
COSTS = {'jsd_upstream': {'jsd': True}, 'ce': {},
         'jsd_standard': {'jsd': 'standard'}, 'euclidean': {'euclidian': True}}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def original_loss():
    baseline = HERE / 'core_integration/ot_loss_before_temporal_smoothing.py'
    expected = json.loads((baseline.parent / 'BASELINE.json').read_text())['sha256']
    assert sha(baseline) == expected, 'Pre-edit source snapshot changed'
    source = baseline.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == 'batched_ottc_loss_bucketized')
    namespace = dict(vars(ot_loss))
    # Actual CUDA allocations: no CPU/device substitution in this script.
    exec(compile(ast.get_source_segment(source, node), str(baseline), 'exec'), namespace)
    return namespace[node.name]


def core_checks(weight, tau):
    old, current = original_loss(), ot_loss.batched_ottc_loss_bucketized
    torch.manual_seed(42)
    x = torch.randn(2, 31, 42, device='cuda', requires_grad=True)
    raw_mass = torch.randn(2, 31, device='cuda', requires_grad=True)
    mask = torch.arange(31, device='cuda')[None] < torch.tensor([31, 19], device='cuda')[:, None]
    alpha = raw_mass.masked_fill(~mask, -torch.inf).softmax(-1)
    labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]], device='cuda')
    y = F.one_hot(labels, 42)
    bm = torch.tensor([[1., 1., 1., 1.], [1., 1., 1., 0.]], device='cuda')
    beta = bm / bm.sum(-1, keepdim=True)
    penalty = ot_loss.temporal_smoothing_loss(x, mask, tau=tau, reduction='sum')
    report = {}
    for name, options in COSTS.items():
        before = old(x, y, alpha, beta, bmask=bm, **options)
        disabled = current(x, y, alpha, beta, bmask=bm, **options)
        enabled = current(x, y, alpha, beta, bmask=bm, smooth_weight=weight,
                          smooth_tau=tau, smooth_mask=mask, **options)
        assert len(before) == len(disabled) == len(enabled) == 4
        for a, b in zip(before, disabled):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(before[1:], enabled[1:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected = before[0] + weight * penalty
        torch.testing.assert_close(enabled[0], expected, rtol=0, atol=0)
        gradients = [torch.autograd.grad(loss, (x, raw_mass), retain_graph=True)
                     for loss in [before[0], disabled[0], enabled[0], expected]]
        for i, j in [(0, 1), (2, 3)]:
            for a, b in zip(gradients[i], gradients[j]):
                # Repeated-index CUDA backward may use atomic accumulation.
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gradients[0][1], gradients[2][1], rtol=1e-5, atol=1e-6)
        assert all(torch.isfinite(g).all() for group in gradients for g in group)
        report[name] = dict(base_loss=float(before[0]), smooth_sum=float(penalty),
            total_loss=float(enabled[0]), default_off_exact=True, transport_unchanged=True,
            gradients_verified=True, alpha_gradient_unchanged=True,
            gradient_comparison_rtol=1e-5, gradient_comparison_atol=1e-6,
            default_gradient_max_abs_difference=max(float((a-b).abs().max()) for a,b in zip(gradients[0],gradients[1])))
        LOG.info('CUDA core %-12s OTTC=%.6f smooth=%.6f total=%.6f PASS',
                 name, float(before[0]), float(penalty), float(enabled[0]))
    return report


def gradient_group(parameters, allow_unused=False):
    parameters = [p for p in parameters if p.requires_grad]
    if allow_unused:
        # SSL encoder LayerDrop can skip layers on a particular training pass.
        parameters = [p for p in parameters if p.grad is not None]
    assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm = sum(float(p.grad.float().square().sum()) for p in parameters) ** .5
    assert norm > 0, 'Expected a nonzero trainable gradient group'
    return norm


def real_batch_probe(args, output):
    import transformers
    transformers.utils.import_utils._peft_available = False
    transformers.utils.import_utils._bitsandbytes_available = False
    from transformers import Wav2Vec2Processor
    from reproduction.timit_ssl_native10_20261019 import common as base

    torch.manual_seed(42)
    config = base.recipe(f'wav2vec2__timit__10ms__{args.head}__{args.cost}')
    weights = base.SSL / 'pytorch_model.bin'
    LOG.info('核对原始SSL权重，然后加载真实 native10ms 模型。')
    assert sha(weights) == base.SSL_SHA
    weights_before = weights.stat()
    processor_dir = ROOT / 'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/processor'
    vocab = json.loads((processor_dir / 'vocab.json').read_text())
    processor = Wav2Vec2Processor.from_pretrained(processor_dir)
    manifest = ROOT / 'reproduction/multihead_timit/data/train-train.jsonl'
    # Use training examples only. Never pass reference boundary times to loss.
    selected = []
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            selected.append(dict(utt=row['utt'], wav=row['wav'],
                                 labels=row['targets']['ottc_arpabet']))
            if len(selected) == args.batch_size:
                break
    assert len(selected) == args.batch_size
    model, topo = base.build_model(config, vocab, 73)
    model.cuda().train()
    batch = base.Collator(processor, topo['mapping'] if topo else None)(selected)
    batch = {key: value.cuda() for key, value in batch.items()}
    labels = batch.pop('labels')
    bm = (labels >= 0).float()
    y = F.one_hot(labels.masked_fill(labels < 0, 0), model.config.vocab_size)
    beta = bm / bm.sum(-1, keepdim=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=.005) if args.steps else None
    cnn = model.wav2vec2.feature_extractor.conv_layers
    assert cnn[-1].conv.stride == (1,)
    frozen_before = cnn[0].conv.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    last_before = cnn[-1].conv.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(max(1, args.steps)):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.monotonic()
        # Audio-only forward, then call the requested ot_loss.py entry directly.
        out = model(**batch)
        logits, alpha, mask = out.logits, out.hidden_states[0], out.hidden_states[1]
        base_result = ot_loss.batched_ottc_loss_bucketized(logits, y, alpha, beta, bmask=bm, **COSTS[args.cost])
        result = ot_loss.batched_ottc_loss_bucketized(logits, y, alpha, beta, bmask=bm,
            smooth_weight=args.weight, smooth_tau=args.tau, smooth_mask=mask, **COSTS[args.cost])
        smooth = ot_loss.temporal_smoothing_loss(logits, mask, tau=args.tau, reduction='sum')
        torch.testing.assert_close(result[0], base_result[0] + args.weight * smooth, rtol=0, atol=0)
        for old, new in zip(base_result[1:], result[1:]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        assert torch.isfinite(result[0])
        (result[0] / args.batch_size).backward()
        gradients = dict(last_cnn=gradient_group(cnn[-1].parameters()),
            encoder=gradient_group(model.wav2vec2.encoder.parameters(), allow_unused=True),
            classification_head=gradient_group(model.lm_head.parameters()),
            alpha_head=gradient_group(model.lm_weight.parameters()))
        assert all(p.grad is None for layer in cnn[:-1] for p in layer.parameters())
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
        torch.cuda.synchronize()
        record = dict(probe_step=step + 1, optimizer_updated=optimizer is not None,
            raw_ottc_sum=float(base_result[0]), raw_smooth_sum=float(smooth),
            weighted_smooth_sum=args.weight * float(smooth), total_sum=float(result[0]),
            total_per_utterance=float(result[0]) / args.batch_size,
            valid_output_frames=[int(v) for v in mask.sum(-1)],
            gradient_norms=gradients, seconds=time.monotonic() - begin,
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
        records.append(record)
        with (output / 'real_batch_steps.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        LOG.info('真实batch step %d: OTTC=%.6f smooth=%.6f λ·smooth=%.6f total=%.6f peak=%.2fGiB',
                 step + 1, record['raw_ottc_sum'], record['raw_smooth_sum'], record['weighted_smooth_sum'],
                 record['total_sum'], record['peak_allocated_GiB'])
    assert torch.equal(frozen_before, cnn[0].conv.weight)
    head_updated = not torch.equal(head_before, model.lm_head.weight)
    last_updated = not torch.equal(last_before, cnn[-1].conv.weight)
    if args.steps:
        assert head_updated and last_updated
    weights_after = weights.stat()
    assert (weights_before.st_ino, weights_before.st_size, weights_before.st_mtime_ns) == (
        weights_after.st_ino, weights_after.st_size, weights_after.st_mtime_ns)
    report = dict(initialization='original SSL encoder; fresh task heads; native10ms', head=args.head,
        not_a_trained_checkpoint_benchmark=True, config=config, training_manifest_sha256=sha(manifest),
        vocabulary_sha256=sha(processor_dir / 'vocab.json'), utterances=[r['utt'] for r in selected],
        batch_size=args.batch_size, optimizer_updates=args.steps, records=records,
        first_six_CNN_layers_frozen=True, first_CNN_unchanged=True,
        last_CNN_updated=last_updated, classification_head_updated=head_updated,
        pretrained_file_unchanged=True, model_checkpoints_written=False, reference_boundary_times_used=False)
    del model, optimizer, out, result, base_result, batch, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=float, default=.1)
    parser.add_argument('--tau', type=float, default=4.)
    parser.add_argument('--steps', type=int, default=1, help='0: backward only; 1..20: bounded real-batch updates')
    parser.add_argument('--batch-size', type=int, default=2, help='1..8 training utterances')
    parser.add_argument('--cost', choices=['jsd_upstream', 'ce', 'jsd_standard'], default='jsd_upstream')
    parser.add_argument('--head', choices=['mono', 's3t2ss'], default='mono')
    parser.add_argument('--core-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.weight) or args.weight < 0 or not math.isfinite(args.tau) or args.tau <= 0:
        parser.error('weight must be finite/nonnegative and tau finite/positive')
    if not 0 <= args.steps <= 20 or not 1 <= args.batch_size <= 8:
        parser.error('steps must be 0..20 and batch-size 1..8 for a bounded probe')
    if not os.environ.get('PBS_JOBID') or not torch.cuda.is_available():
        parser.error('Run inside a PBS GPU allocation with CUDA available; see interactive_gpu.sh --help')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir or HERE / 'gpu_probes' / f'{stamp}_{os.getpid()}'
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output-dir must be new or empty; previous probe results are preserved')
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(output / 'probe.log')])
    tracked = [Path(__file__), HERE / 'interactive_gpu.sh', ROOT / 'ottc/ops_loss/ot_loss.py',
               ROOT / 'ottc/ops_loss/temporal_smoothing.py']
    sources = {str(path): sha(path) for path in tracked}
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        output_dir=str(output), host=platform.node(), pbs_job=os.environ['PBS_JOBID'],
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        precision='FP32', source_sha256=sources, evaluation_uploads=False,
        scope='bounded synthetic core checks and optional real training-batch probe; not production training')
    write(output / 'CONFIG.json', config)
    begin = time.monotonic()
    LOG.info('GPU=%s PBS=%s 输出=%s', config['gpu'], config['pbs_job'], output)
    try:
        core = core_checks(args.weight, args.tau)
        write(output / 'CORE_CHECKS.json', core)
        real = None if args.core_only else real_batch_probe(args, output)
        if real is not None:
            write(output / 'REAL_BATCH.json', real)
        assert all(sha(path) == value for path, value in sources.items()), 'Source changed during probe'
        write(output / 'PROBE_COMPLETE.json', dict(status='passed', config=config, core_checks=core,
            real_batch_probe=real, seconds=time.monotonic() - begin, source_files_unchanged=True))
        LOG.info('PASS：全部检查通过。完整结果：%s', output / 'PROBE_COMPLETE.json')
    except Exception as error:
        write(output / 'FAILED.json', dict(status='failed', error=repr(error), seconds=time.monotonic() - begin))
        LOG.exception('Probe failed; local logs retained at %s', output)
        raise


if __name__ == '__main__':
    main()
