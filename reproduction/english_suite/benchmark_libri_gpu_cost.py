"""Bounded fixed-batch Libri960 compute and GPU telemetry comparison."""
import argparse
import gc
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import wandb

ROOT = Path('/work/gm64/m64000/OTTC')
import sys
sys.path.insert(0, str(ROOT/'reproduction'))
from train_multihead import AudioTargets, MultiPhoneModel, collate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    from reproduction.wandb_policy import init_evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=7)
    args = parser.parse_args()
    assert args.steps >= 3 and torch.cuda.is_available()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_script = args.output/'source/summarize_gpu_cost.py'
    shutil.copy2(ROOT/'reproduction/english_suite/summarize_gpu_cost.py', summary_script)
    manifest = ROOT/'reproduction/multihead_libri960/data/train-train.jsonl'
    data = AudioTargets(manifest)
    ranked = sorted(range(len(data)), key=lambda i: data.records[i]['duration'])
    offset = len(ranked)//2 - 32
    ids = {'typical': ranked[offset:offset+64], 'long': ranked[-64:]}
    samples = {kind: [data[i] for i in selected] for kind, selected in ids.items()}
    write_json(args.output/'fixtures.json', {'manifest': str(manifest), 'sha256': sha256(manifest),
                                           'record_indices': ids,
                                           'records': {k: [s['record'] for s in batch] for k, batch in samples.items()}})
    plan = json.loads((args.output/'plan.json').read_text())
    options = dict(project='ottc-english-multiphoneset', entity='kevingenghaopeng-',
                   group='gpu_cost_probe', name=f'libri_gpu_cost_{plan["job"]}', job_type='profiling',
                   dir=str(args.output), config={'effective_batch': 64, 'dataset': 'Libri960',
                                                 'production_updated': False, 'plan': plan},
                   settings=wandb.Settings(init_timeout=20, disable_code=True))
    try:
        run = init_evaluation(**options)
    except wandb.errors.CommError:
        run = init_evaluation(**options, mode='offline')
    write_json(args.output/'wandb_run.json', {'id': run.id, 'url': run.url})
    results = []
    for objective in ('ctc', 'ce', 'jsd_upstream'):
        checkpoint = Path(plan['checkpoints'][objective]['snapshot'])
        model = MultiPhoneModel.from_pretrained(checkpoint).cuda().train()
        if model.config.freeze_alignment:
            model.freeze_alignment()
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for kind, batch in samples.items():
            audio_seconds = sum(s['record']['duration'] for s in batch)
            for microbatch, checkpointing in ((8, True), (16, True), (32, True),
                                              (8, False), (16, False), (32, False)):
                model.load_state_dict(initial)
                if checkpointing:
                    model.gradient_checkpointing_enable()
                else:
                    model.gradient_checkpointing_disable()
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                             lr=5e-5 if objective == 'ctc' else 2e-4)
                records, error = [], None
                cpu = inputs = loss = None
                try:
                    for step in range(args.steps):
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        stamp = time.time()
                        started = time.perf_counter()
                        loss_sum = 0.
                        for first in range(0, 64, microbatch):
                            cpu = collate(batch[first:first+microbatch])
                            inputs = {key: {h: v.cuda() for h, v in value.items()} if isinstance(value, dict)
                                      else value.cuda() for key, value in cpu.items()}
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                loss = model(**inputs)['loss']/(64//microbatch)
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError('Nonfinite loss')
                            loss.backward()
                            loss_sum += float(loss.detach())
                            cpu = inputs = loss = None
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                        records.append({'step': step, 'start_unix': stamp, 'end_unix': time.time(),
                                        'seconds': time.perf_counter()-started, 'loss': loss_sum,
                                        'grad_norm': float(norm),
                                        'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                                        'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30})
                except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
                    error = str(exc).split('\n')[0]
                cpu = inputs = loss = None
                optimizer.zero_grad(set_to_none=True)
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
                entry = {'objective': objective, 'fixture': kind, 'microbatch': microbatch,
                         'gradient_checkpointing': checkpointing, 'alignment_frozen': model.config.freeze_alignment,
                         'effective_batch': 64, 'status': 'FAIL' if error else 'PASS', 'error': error,
                         'checkpoint': str(checkpoint), 'gpu': torch.cuda.get_device_name(),
                         'capacity_GiB': torch.cuda.get_device_properties(0).total_memory/2**30,
                         'records': records, 'production_updated': False}
                if not error:
                    seconds = sum(r['seconds'] for r in records[1:])/(args.steps-1)
                    entry.update(mean_step_seconds=seconds, audio_seconds_per_wall_second=audio_seconds/seconds,
                                 gpu_seconds_per_audio_hour=seconds/audio_seconds*3600,
                                 peak_GiB=max(r['peak_allocated_GiB'] for r in records))
                results.append(entry)
                write_json(args.output/'results.json', results)
                print(json.dumps(entry), flush=True)
                run.log({k: v for k, v in entry.items() if k != 'records'})
                for record in records:
                    run.log({'objective': objective, 'fixture': kind, 'microbatch': microbatch,
                             'gradient_checkpointing': checkpointing, **{'train/'+k: v for k, v in record.items()}})
        del model, initial
        gc.collect()
        torch.cuda.empty_cache()
    run.finish()
    baseline_ok = all(r['status'] == 'PASS' for r in results
                      if r['microbatch'] == 8 and r['gradient_checkpointing'])
    write_json(args.output/'BENCHMARK_COMPLETE.json', {'passed': baseline_ok, 'candidates': len(results),
                                                     'production_updated': False,
                                                     'scope': 'fixed Libri960 batches; collate/transfer/loss/backward/clip/AdamW; excludes audio reads, dev/test'})
    subprocess.run([sys.executable, str(summary_script), '--libri-dir', str(args.output),
                    '--mfa-dir', str(ROOT/'reproduction/english_suite/gpu_cost_probe_3512870.opbs'),
                    '--output', str(args.output/'summary.md')], check=True)


if __name__ == '__main__':
    main()
