"""Profile the actual full40 workload, including longest batches and short tail."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import set_seed
from reproduction.buckeye_frame_ag_20261027 import common
from reproduction.timit_frame_divergences_20261026.profile_recipe import gradient_checks
from reproduction.timit_ssl_native10_20261019.validate import verify_ssl_encoder


def candidate(config, data, rows, vocab, processor, output, micro):
    set_seed(config['seed'])
    assert common.sha(common.SSL / 'pytorch_model.bin') == common.SSL_SHA
    model, topo = common.build_model(config, vocab, math.ceil(len(data['train']) / 64))
    assert topo is None and all(p.dtype == torch.float32 for p in model.parameters())
    initialization = verify_ssl_encoder(model)
    model.cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collator = common.Collator(processor)
    longest = sorted(range(len(rows['train'])), key=lambda i: rows['train'][i]['duration'], reverse=True)
    typical = np.random.default_rng(42).permutation(len(rows['train'])).tolist()
    tail = len(rows['train']) % 64
    records = []
    for label, indices in [('longest64', longest[:64]), ('representative64', typical[:64]),
                           ('tail45', typical[:tail])]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        loss_value = 0.
        for start in range(0, len(indices), micro):
            batch = collator([data['train'][int(i)] for i in indices[start:start+micro]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**{k: v.cuda() for k, v in batch.items()}).loss / len(indices)
            assert torch.isfinite(loss)
            loss.backward()
            loss_value += float(loss.detach())
            del batch, loss
        gradients = gradient_checks(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.monotonic() - begin
        records.append(dict(case=label, logical_batch=len(indices), microbatch=micro,
            seconds=seconds, loss=loss_value, gradients=gradients,
            peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
            audio_seconds_per_second=sum(rows['train'][int(i)]['duration'] for i in indices)/seconds))
        print(json.dumps(records[-1]), flush=True)
    # Longest real validation batch, then a training backward after validation.
    model.eval()
    indices = sorted(range(len(rows['dev'])), key=lambda i: rows['dev'][i]['duration'], reverse=True)[:micro]
    batch = collator([data['dev'][i] for i in indices])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        valid = model(**{k: v.cuda() for k, v in batch.items()})
        assert torch.isfinite(valid.loss)
    del batch, valid
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = collator([data['train'][int(i)] for i in typical[:min(micro, 16)]])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**{k: v.cuda() for k, v in batch.items()}).loss / min(micro, 16)
    loss.backward()
    gradient_checks(model)
    del batch, loss
    from reproduction.buckeye_frame_ag_20261027.evaluate import evaluate
    probe = evaluate(config, model, rows['dev'][:4], vocab, processor, None,
                     output / f'evaluation_probe_{micro}.json')
    assert probe['modes']['forced_alignment']['alignment_failures'] == 0
    path = output / f'profile_weights_{micro}.pt'
    common.atomic_weights(model, path, dict(recipe=config['name'], epoch=0, step=0, profile_only=True))
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
    model.load_state_dict(saved['model'])
    result = dict(passed=True, microbatch=micro, accumulation_steps=math.ceil(64/micro),
        batches=records, config=config, source_sha256=common.source_hashes(),
        initial_SSL_verified=initialization, saved_model_dtype='float32',
        validation_to_training_verified=True, saved_model_reload_verified=True,
        host=platform.node(), job=os.environ.get('PBS_JOBID'))
    common.write(output / f'CANDIDATE_{micro}.json', result)
    del saved, model, optimizer
    path.unlink()
    gc.collect()
    torch.cuda.empty_cache()


def run_profile(config, data, rows, vocab, processor, output):
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    # Independent subprocesses prevent an OOM from contaminating the next probe.
    for micro in (64, 48):
        log = output / f'candidate_{micro}.log'
        with log.open('w') as stream:
            completed = subprocess.run([sys.executable, '-u', '-m',
                'reproduction.buckeye_frame_ag_20261027.profile_recipe', '--micro', str(micro)],
                stdout=stream, stderr=subprocess.STDOUT)
        attempts.append(dict(microbatch=micro, exit_code=completed.returncode, log=str(log)))
        if completed.returncode not in (0, 42):
            raise RuntimeError(f'Profile failed: {log}; exit={completed.returncode}')
    passed = [json.loads((output / f'CANDIDATE_{a["microbatch"]}.json').read_text())
              for a in attempts if a['exit_code'] == 0]
    if not passed:
        raise RuntimeError('Both physical batches failed GPU profile; no training authorized by gate')
    # End-to-end representative logical-batch time includes loading, backward,
    # clipping and optimizer; longest64 and tail must pass for either candidate.
    best = min(passed, key=lambda r: next(b['seconds'] for b in r['batches'] if b['case']=='representative64'))
    common.write(output / 'PASS.json', dict(best, attempts=attempts,
        selection='fastest real representative64 end-to-end time among longest64/tail/validation-safe candidates'))
    print('GPU_PROFILE_PASS', best['microbatch'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro', type=int, choices=(48, 64), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = common.recipe(common.NAME)
    data, rows, vocab, processor, audit = common.load_data(config)
    config['data'] = audit
    output = common.HERE / 'profiles' / common.NAME
    output.mkdir(parents=True, exist_ok=True)
    try:
        candidate(config, data, rows, vocab, processor, output, args.micro)
    except torch.OutOfMemoryError as error:
        common.write(output / f'OOM_{args.micro}.json', dict(microbatch=args.micro, error=str(error)))
        raise SystemExit(42)


if __name__ == '__main__':
    main()
