"""Bounded, local-only CUDA checks and real-TIMIT forward/backward comparisons."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def disable_stochasticity(torch, model):
    model.wav2vec2.config.apply_spec_augment = False
    model.wav2vec2.config.layerdrop = 0.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if isinstance(getattr(module, 'dropout', None), (float, int)):
            module.dropout = 0.


def measure(torch, model, batch, amp, repeats):
    model.cuda().train()
    measurements = []
    for iteration in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            output = model(**batch)
        output.loss.backward()
        torch.cuda.synchronize()
        record = dict(seconds=time.perf_counter() - started, loss=float(output.loss.detach()),
            peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30,
            incremental_peak_GiB=(torch.cuda.max_memory_allocated() - initial) / 2 ** 30)
        assert torch.isfinite(output.loss), 'nonfinite loss'
        if iteration:
            measurements.append(record)
        del output
    loss_spread = max(row['loss'] for row in measurements) - min(row['loss'] for row in measurements)
    assert loss_spread <= 1e-5, f'nondeterministic reference/probe losses: {loss_spread}'
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all()), 'nonfinite gradient: ' + name
            gradients[name] = parameter.grad.detach().cpu().clone()
    assert gradients and bool(gradients['lm_head.weight'].abs().sum() > 0)
    assert bool(gradients['lm_weight.weight'].abs().sum() > 0)
    selected = getattr(model, 'last_pruned_vocabulary', None)
    selected_ids = None if selected is None else selected.selected.cpu().tolist()
    model.zero_grad(set_to_none=True)
    model.cpu()
    if hasattr(model, 'last_pruned_vocabulary'):
        model.last_pruned_vocabulary = None
    torch.cuda.empty_cache()
    return dict(measurements=measurements, median_seconds=statistics.median(row['seconds'] for row in measurements),
        repeated_loss_spread=loss_spread,
        peak_allocated_GiB=max(row['peak_allocated_GiB'] for row in measurements),
        incremental_peak_GiB=max(row['incremental_peak_GiB'] for row in measurements),
        selected_global_ids=selected_ids), gradients


def compare_gradients(torch, actual, expected, amp):
    assert actual.keys() == expected.keys(), 'gradient parameter sets differ'
    relative_tolerance, absolute_tolerance = (.05, .001) if amp else (.002, .00002)
    results = {}
    passed = True
    for name, reference in expected.items():
        difference = (actual[name].float() - reference.float()).abs()
        within = bool((difference <= absolute_tolerance + relative_tolerance * reference.float().abs()).all())
        passed = passed and within
        results[name] = dict(passed=within, max_absolute_error=float(difference.max()),
            relative_l2_error=float(difference.norm() / reference.float().norm().clamp_min(1e-12)),
            reference_norm=float(reference.float().norm()))
    return dict(passed=passed, rtol=relative_tolerance, atol=absolute_tolerance, parameters=results)


def run(args, output):
    import torch
    import transformers
    transformers.utils.import_utils._peft_available = False
    transformers.utils.import_utils._bitsandbytes_available = False
    from transformers import Wav2Vec2FeatureExtractor
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
    from reproduction.native_stride_ottc_20261018.common import Collator
    from reproduction.pruned_ottc_20261020.check import main as check
    from reproduction.pruned_ottc_20261020.model import PrunedOTTCMixin

    assert torch.cuda.is_available() and os.environ.get('PBS_JOBID'), 'allocated PBS GPU required'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check(['--device', 'cuda', '--output', str(output / 'cuda_checks.json')])
    torch.set_num_threads(4)
    data = ROOT / 'reproduction/timit_bpe500_20261020/data'
    metadata = json.loads((data / 'metadata.json').read_text())
    manifest = data / 'train.jsonl'
    assert sha256(manifest) == metadata['manifests_sha256']['train']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    batches = {'first_two': rows[:2], 'longest_two': sorted(rows, key=lambda row: row['duration'], reverse=True)[:2]}
    backbone = ROOT / 'reproduction/models/wav2vec2-large-lv60'
    processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
        do_normalize=True, return_attention_mask=True)
    protocol = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), total_GPU_GiB=torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
        data_metadata=metadata, backbone_sha256=sha256(backbone / 'pytorch_model.bin'),
        backbone_config_sha256=sha256(backbone / 'config.json'), seed=42, repeats=args.repeats,
        chunk_size=args.chunk_size, optimizer_updates=0, evaluation_uploads=False,
        precision='Both heads project in FP32; BF16 applies to encoder. Not legacy BF16-rounded dense-head timing.',
        stochasticity='Module and functional-attention dropout, layerdrop and SpecAugment disabled; repeated loss checked.',
        timing='One unmeasured warmup, CUDA-synchronized forward + backward; excludes audio loading and gradient export.',
        batches={name: [dict(utt=row['utt'], duration=row['duration'], labels=row['labels'],
                            wav_sha256=sha256(Path(row['wav']))) for row in items] for name, items in batches.items()})
    write_json(output / 'PROTOCOL.json', protocol)

    class FP32Head(torch.nn.Linear):
        def forward(self, hidden):
            with torch.autocast(hidden.device.type, enabled=False):
                return torch.nn.functional.linear(hidden.float(), self.weight.float(),
                    None if self.bias is None else self.bias.float())

    class DenseModel(wvForOTTC):
        def __init__(self, config):
            super().__init__(config)
            replacement = FP32Head(self.lm_head.in_features, self.lm_head.out_features,
                                   bias=self.lm_head.bias is not None)
            replacement.load_state_dict(self.lm_head.state_dict())
            self.lm_head = replacement

    class PrunedModel(PrunedOTTCMixin, DenseModel):
        pass

    torch.manual_seed(42)
    dense = DenseModel(wvOTTCConfig(wav2vec2_path=str(backbone), vocab_size=500,
        hidden_size=1024, bins_metric='jsd', forward_type='regular', pad_token_id=0))
    pruned = PrunedModel(wvOTTCConfig(wav2vec2_path=str(backbone), vocab_size=500,
        hidden_size=1024, bins_metric='jsd', forward_type='regular', pad_token_id=0),
        pruned_chunk_size=args.chunk_size)
    pruned.load_state_dict(dense.state_dict(), strict=True)
    for model in (dense, pruned):
        model.freeze_feature_encoder()
        disable_stochasticity(torch, model)
    records = []
    for batch_name, items in batches.items():
        batch = {name: value.cuda() for name, value in Collator(processor)(items).items()}
        for cost in ('ce', 'jsd'):
            dense.jsd = pruned.jsd = cost == 'jsd'
            for amp in (False, True):
                case = dict(batch=batch_name, cost=cost, precision='bf16_encoder' if amp else 'fp32',
                            batch_size=len(items), input_samples=list(batch['input_values'].shape))
                print('CASE', json.dumps(case), flush=True)
                dense_result, dense_gradients = measure(torch, dense, batch, amp, args.repeats)
                pruned_result, pruned_gradients = measure(torch, pruned, batch, amp, args.repeats)
                gradients = compare_gradients(torch, pruned_gradients, dense_gradients, amp)
                difference = abs(pruned_result['measurements'][-1]['loss'] - dense_result['measurements'][-1]['loss'])
                reference_loss = abs(dense_result['measurements'][-1]['loss'])
                tolerance = .0002 if amp else .00002
                case.update(dense=dense_result, pruned=pruned_result, gradients=gradients,
                    loss_absolute_error=difference, loss_tolerance=dict(rtol=tolerance, atol=tolerance),
                    passed=gradients['passed'] and difference <= tolerance * (1 + reference_loss),
                    peak_memory_ratio=pruned_result['peak_allocated_GiB'] / dense_result['peak_allocated_GiB'],
                    runtime_ratio=pruned_result['median_seconds'] / dense_result['median_seconds'])
                records.append(case)
                write_json(output / 'RESULTS.json', dict(passed=all(row['passed'] for row in records),
                    complete=False, evaluation_uploads=False, cases=records))
                print('RESULT', json.dumps({key: case[key] for key in
                    ('batch', 'cost', 'precision', 'passed', 'loss_absolute_error', 'peak_memory_ratio', 'runtime_ratio')}), flush=True)
                del dense_gradients, pruned_gradients
                gc.collect()
                assert case['passed'], 'real-batch equivalence failed; see RESULTS.json'
        del batch
    write_json(output / 'RESULTS.json', dict(passed=True, complete=True, evaluation_uploads=False, cases=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    assert args.chunk_size > 0 and args.repeats > 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [*HERE.glob('*.py'), *HERE.glob('*.pbs'),
        ROOT / 'ottc/models/sequence/wav2vec2_ottc.py', ROOT / 'ottc/ops_loss/ot_loss.py',
        ROOT / 'reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py',
        ROOT / 'reproduction/native_stride_ottc_20261018/common.py',
        ROOT / 'reproduction/libri960_continue5_20261017/environment.sh']
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    for path in paths:
        destination = output / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(output / 'RUN.json', dict(job=os.environ.get('PBS_JOBID'), source_sha256=sources,
        chunk_size=args.chunk_size, repeats=args.repeats, evaluation_uploads=False, started_unix=time.time()))
    try:
        run(args, output)
        assert all(sha256(ROOT / path) == digest for path, digest in sources.items()), 'sources changed during probe'
        write_json(output / 'PASS.json', dict(passed=True, evaluation_uploads=False, source_sha256=sources))
    except BaseException as error:
        write_json(output / 'FAIL.json', dict(passed=False, error=repr(error), traceback=traceback.format_exc(),
            evaluation_uploads=False, source_sha256=sources))
        raise


if __name__ == '__main__':
    main()