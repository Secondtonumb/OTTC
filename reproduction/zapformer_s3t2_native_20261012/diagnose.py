"""Small GPU probes separating native arithmetic repeatability from persistence."""
import runtime
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import train as native
from task_model import S3T2
from data import Inputs, jsd_rows, plan
from augmentation import augment

HERE = Path(__file__).resolve().parent


def differences(left, right):
    result = {}
    for key, x in left.items():
        y = right[key]
        if x is None or y is None:
            if x is not y:
                result[key] = 'missing gradient'
        elif not torch.equal(x, y):
            result[key] = float((x - y).abs().max())
    return result


def worker(output, strict, deterministic_k2):
    torch.set_num_threads(4)
    torch.autograd.set_multithreading_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_k2:
        from k2_determinism import enable
        enable()
    if strict:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    net = S3T2(tiny=True).cuda().train()
    optimizer = native.BatchedRubik(native.get_parameter_groups_with_lrs(net, lr=.02), lr=.02, beta1=.99)
    rows = jsd_rows(64)
    batch = Inputs(rows).batch(plan(rows, 10, 0, 1, 0)[0])
    native.fix_random_seed(42); torch.cuda.manual_seed(92)
    specaug = native.AlternatingSpecAugment(seed=92)
    features, _ = augment(specaug, batch['inputs'].cuda(), batch['supervisions']['num_frames'].cuda())
    initial = {name: p.detach().cpu().clone() for name, p in net.named_parameters()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = net(features, batch['supervisions']['num_frames'].cuda(), batch['tokens'].cuda(), batch['token_lengths'].cuda(), aux_loss_scale=.05)
    loss.backward()
    grads = {name: p.grad.detach().cpu().clone() if p.grad is not None else None for name, p in net.named_parameters()}
    optimizer.step()
    updated = {name: p.detach().cpu().clone() for name, p in net.named_parameters()}
    torch.save(dict(initial=initial, gradients=grads, updated=updated, loss=float(loss.detach()),
                    optimizer=optimizer.state_dict(), version=torch.__version__, gpu=torch.cuda.get_device_name()), output)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--worker', type=Path); ap.add_argument('--strict', action='store_true')
    ap.add_argument('--deterministic-k2', action='store_true')
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, a.strict, a.deterministic_k2); return
    out = HERE/('diagnostics_k2' if a.deterministic_k2 else 'diagnostics_blocking' if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1' else 'diagnostics'); out.mkdir(exist_ok=True)
    results = {}
    for strict in (False, True):
        states = []
        for repeat in range(2):
            path = out/f'precision-{strict}-{repeat}.pt'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(path)]
            if strict: command.append('--strict')
            if a.deterministic_k2: command.append('--deterministic-k2')
            subprocess.run(command, check=True)
            states.append(torch.load(path, map_location='cpu', weights_only=False))
        x, y = states
        results[strict] = dict(loss=[x['loss'],y['loss']], version=x['version'], gpu=x['gpu'],
                              **{key:differences(x[key], y[key]) for key in ('initial','gradients','updated')})
    (out/'repeatability.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__': main()
