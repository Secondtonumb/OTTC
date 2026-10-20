"""Compare physical GPU counts on identical global MFA batches; never resume training."""
import argparse
import gc
import gzip
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import make_optimizer
from film_k4_10ms import FiLMK4
from model_10ms import MFAZapformer, parameters
from step_checkpoint import tree_to
import microbatching

BASE = Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922')
SUITE = Path('/work/gm64/m64000/OTTC/reproduction/english_suite')
DATA = Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json')
ARMS = {'ctc': (315, 4, 26266), 'ottc': (315, 4, 24287),
        'spottc': (320, 2, 22643), 'film_k4': (200, 4, 0)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def fixture_rows(output, steps):
    fixtures, wanted = {}, set()
    provenance = {'manifest': str(DATA), 'manifest_sha256': sha256(DATA), 'fixtures': {}}
    for arm in ARMS:
        source = BASE/'batch_probe_3418596.opbs'/f'{arm}_fixtures.json'
        fixture = json.loads(source.read_text())
        slots = [rank_steps[:steps] for rank_steps in fixture['rank_ids']]
        assert len(slots) == 8 and all(len(s) == steps for s in slots)
        for step in range(steps):
            ids = [item for slot in slots for item in slot[step]]
            assert len(ids) == len(set(ids))
            wanted.update(ids)
        wanted.add(fixture['longest'])
        fixtures[arm] = {'slots': slots, 'longest': fixture['longest']}
        provenance['fixtures'][arm] = {'path': str(source), 'sha256': sha256(source)}
    meta = json.loads(DATA.read_text())
    entry = meta['split_manifests'][meta['train_split']]
    source = DATA.parent/entry['path']
    assert sha256(source) == entry['sha256']
    lookup, count = {}, 0
    with gzip.open(source, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if row['id'] in wanted:
                lookup[row['id']] = row
    assert count == entry['count'] and wanted == lookup.keys()
    payload = {'fixtures': fixtures, 'rows': lookup, 'metadata': meta, 'provenance': provenance}
    write_json(output/'fixtures.json', payload)
    return payload


def restore(net, initial, saved, arm, rank):
    net.load_state_dict(initial)
    settings = parameters()
    settings.base_lr = .02
    optimizer = make_optimizer(net, settings)
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        for live, old in zip(optimizer.param_groups, saved['optimizer']['param_groups']):
            for parameter, index in zip(live['params'], old['params']):
                if index in saved['optimizer']['state']:
                    optimizer.state[parameter] = tree_to(saved['optimizer']['state'][index], parameter.device)
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    return optimizer


def main():
    from reproduction.wandb_policy import init_evaluation
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    assert 0 < args.warmup < args.steps <= 10
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The coordinator uses CPU collectives so inactive GPUs do not spin in NCCL.
    dist.init_process_group('gloo', timeout=timedelta(minutes=16))
    assert dist.get_world_size() == 8
    args.output.mkdir(parents=True, exist_ok=True)
    payload = [fixture_rows(args.output, args.steps) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    data = payload[0]
    groups = {n: dist.new_group(list(range(n)), backend='nccl', timeout=timedelta(seconds=120))
              for n in (1, 2, 4, 8)}
    maps = {}

    def load(selected):
        torch.cuda.set_device(0)
        started = time.perf_counter()
        xs = []
        for row in selected:
            path = DATA.parent/row['feature_file']
            key = str(path)
            if key not in maps:
                maps[key] = np.memmap(path, mode='r', dtype=np.float32)
            first = row['feature_offset']
            array = maps[key][first:first + 80*row['frames']]
            xs.append(torch.from_numpy(np.array(array).reshape(-1, 80)))
        cpu = (torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).pin_memory(),
               torch.tensor([r['frames'] for r in selected]).pin_memory(),
               torch.tensor([t for r in selected for t in r['tokens']]).pin_memory(),
               torch.tensor([len(r['tokens']) for r in selected]).pin_memory())
        return cpu, time.perf_counter() - started

    results, checkpoints = [], {}
    run = None
    if rank == 0:
        import wandb
        settings = dict(project='ottc-mfa-english', entity='kevingenghaopeng-',
                        group='gpu_cost_probe', name=f'mfa_gpu_cost_{os.environ.get("PBS_JOBID")}',
                        job_type='profiling', dir=str(args.output),
                        config={'gpu_counts': [1, 2, 4, 8], 'global_batch': 'identical IDs at every step',
                                'steps': args.steps, 'warmup': args.warmup, 'repeats': 2,
                                'production_updated': False, 'resume_migration_validated': False,
                                'job': os.environ.get('PBS_JOBID'), 'provenance': data['provenance']},
                        settings=wandb.Settings(init_timeout=20, disable_code=True))
        try:
            run = init_evaluation(**settings)
        except wandb.errors.CommError:
            run = init_evaluation(**settings, mode='offline')
        write_json(args.output/'wandb_run.json', {'id': run.id, 'url': run.url})

    for arm, (cap, factor, initial_step) in ARMS.items():
        checkpoint = args.output/'checkpoints'/f'{arm}.pt'
        if initial_step and rank == 0:
            source = SUITE/'mfa_models'/arm/f'step-{initial_step:09d}.pt'
            checkpoint.parent.mkdir(exist_ok=True)
            shutil.copy2(source, checkpoint)
            checkpoints[arm] = {'source': str(source), 'snapshot': str(checkpoint), 'sha256': sha256(checkpoint)}
            assert checkpoints[arm]['sha256'] == sha256(source)
            write_json(args.output/'checkpoint_provenance.json', checkpoints)
        dist.barrier()
        torch.manual_seed(42)
        net = (FiLMK4() if arm == 'film_k4' else MFAZapformer(arm)).cuda().train()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False) if initial_step else None
        initial = saved['model'] if saved else tree_to(net.state_dict(), 'cpu')
        original = microbatching.LIMITS[arm]
        microbatching.LIMITS[arm] = tuple(v*factor for v in original)
        slots = data['fixtures'][arm]['slots']
        lookup = data['rows']
        global_ids = [[item for slot in slots for item in slot[s]] for s in range(args.steps)]
        for repetition, order in enumerate(((8, 4, 2, 1), (1, 2, 4, 8))):
            for count in order:
                dist.barrier()
                group = groups[count]
                result = None
                if rank < count:
                    optimizer = restore(net, initial, saved, arm, rank)
                    batches = [[lookup[item] for slot in range(rank, 8, count) for item in slots[slot][s]]
                               for s in range(args.steps)]
                    gathered = [None]*count
                    dist.all_gather_object(gathered, [[r['id'] for r in b] for b in batches], group=group)
                    for s in range(args.steps):
                        actual = [item for peer in gathered for item in peer[s]]
                        assert len(actual) == len(set(actual)) and set(actual) == set(global_ids[s])
                    del gathered
                    plans = [microbatching.plan([r['frames'] for r in b], arm) for b in batches]
                    # Test capacity without DDP so a local OOM cannot strand a peer in backward.
                    ok, error = True, None
                    stress_peak = 0
                    cpu = tensors = value = raw = loss = None
                    try:
                        stress = max(batches, key=lambda b: max(r['frames'] for r in b))
                        for selected in (stress, [lookup[data['fixtures'][arm]['longest']]]):
                            cpu, _ = load(selected)
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.reset_peak_memory_stats()
                            for indices in microbatching.plan([r['frames'] for r in selected], arm):
                                tensors = tuple(t.cuda(non_blocking=True) for t in microbatching.tensors(cpu, indices))
                                with torch.autocast('cuda', dtype=torch.bfloat16):
                                    value = net(*tensors)
                                raw = value['loss']*len(indices) if arm == 'film_k4' else value[0]
                                loss = raw/len(selected)
                                if not bool(torch.isfinite(loss)):
                                    raise FloatingPointError('Nonfinite stress loss')
                                loss.backward()
                                tensors = value = raw = loss = None
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
                            optimizer.step()
                            torch.cuda.synchronize()
                            stress_peak = max(stress_peak, torch.cuda.max_memory_allocated())
                            cpu = None
                    except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
                        ok, error = False, str(exc).split('\n')[0]
                    cpu = tensors = value = raw = loss = None
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    valid = torch.tensor(int(ok), device='cuda')
                    dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
                    errors = [None]*count
                    dist.all_gather_object(errors, error, group=group)
                    if not bool(valid):
                        result = {'arm': arm, 'gpus': count, 'repetition': repetition,
                                  'status': 'FAIL', 'errors': errors}
                    else:
                        optimizer = restore(net, initial, saved, arm, rank)
                        model = DDP(net, device_ids=[0], broadcast_buffers=False, process_group=group)
                        records = []
                        with ThreadPoolExecutor(max_workers=4) as pool:
                            pending = [pool.submit(load, b) for b in batches]
                            for s, selected in enumerate(batches):
                                dist.barrier(group=group)
                                torch.cuda.synchronize()
                                stamp = time.time()
                                started = time.perf_counter()
                                cpu, load_seconds = pending[s].result()
                                wait = time.perf_counter() - started
                                for module in net.modules():
                                    if hasattr(module, 'batch_count'):
                                        module.batch_count = initial_step + s
                                optimizer.zero_grad(set_to_none=True)
                                torch.cuda.reset_peak_memory_stats()
                                total = torch.tensor(len(selected), device='cuda', dtype=torch.float32)
                                dist.all_reduce(total, group=group)
                                local_sum = 0.
                                for index, indices in enumerate(plans[s]):
                                    context = nullcontext() if index == len(plans[s])-1 else model.no_sync()
                                    with context:
                                        tensors = tuple(t.cuda(non_blocking=True) for t in microbatching.tensors(cpu, indices))
                                        with torch.autocast('cuda', dtype=torch.bfloat16):
                                            value = model(*tensors)
                                        raw = value['loss']*len(indices) if arm == 'film_k4' else value[0]
                                        loss = raw*count/total
                                        if not bool(torch.isfinite(loss)):
                                            raise FloatingPointError('Nonfinite timed loss')
                                        loss.backward()
                                        local_sum += float(raw.detach())
                                        tensors = value = raw = loss = None
                                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
                                optimizer.step()
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter() - started
                                metrics = torch.tensor([elapsed, wait, load_seconds, torch.cuda.max_memory_allocated(),
                                                        torch.cuda.max_memory_reserved(), stress_peak],
                                                       device='cuda', dtype=torch.float64)
                                dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
                                summed = torch.tensor([local_sum, sum(r['duration'] for r in selected)],
                                                      device='cuda', dtype=torch.float64)
                                dist.all_reduce(summed, group=group)
                                record = {'step': s, 'start_unix': stamp, 'end_unix': time.time(),
                                          'seconds': metrics[0].item(), 'input_wait_max': metrics[1].item(),
                                          'cpu_load_seconds_max': metrics[2].item(), 'peak_allocated': metrics[3].item(),
                                          'peak_reserved': metrics[4].item(), 'stress_peak': metrics[5].item(),
                                          'loss': (summed[0]/total).item(), 'audio_seconds': summed[1].item(),
                                          'samples': int(total.item()), 'grad_norm_rank0': float(norm),
                                          'microbatches_rank0': len(plans[s])}
                                records.append(record)
                                cpu = None
                        steady = records[args.warmup:]
                        seconds = sum(r['seconds'] for r in steady)
                        audio = sum(r['audio_seconds'] for r in steady)
                        result = {'arm': arm, 'gpus': count, 'repetition': repetition, 'status': 'PASS',
                                  'logical_max_duration_per_virtual_rank': cap, 'microbatch_factor': factor,
                                  'mean_step_seconds': seconds/len(steady), 'audio_seconds_per_wall_second': audio/seconds,
                                  'gpu_seconds_per_audio_hour': count*seconds/audio*3600,
                                  'peak_GiB': max(max(r['peak_allocated'], r['stress_peak']) for r in records)/2**30,
                                  'gpu': torch.cuda.get_device_name(), 'records': records,
                                  'same_global_ids_verified': True, 'production_updated': False}
                        optimizer.zero_grad(set_to_none=True)
                        del model, optimizer
                report = [result if rank == 0 else None]
                dist.broadcast_object_list(report, src=0)
                if rank == 0:
                    results.append(report[0])
                    print(json.dumps(report[0]), flush=True)
                    write_json(args.output/'results.json', results)
                    run.log({k: v for k, v in report[0].items() if k != 'records'})
                    for record in report[0].get('records', []):
                        run.log({'arm': arm, 'gpus': count, 'repetition': repetition,
                                 **{'train/'+k: v for k, v in record.items()}})
                gc.collect()
                torch.cuda.empty_cache()
        microbatching.LIMITS[arm] = original
        del net, initial, saved
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        passed = len(results) == 32 and all(r['status'] == 'PASS' for r in results)
        write_json(args.output/'BENCHMARK_COMPLETE.json', {'passed': passed, 'production_updated': False,
                                                         'checkpoint_migration_validated': False})
        run.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
