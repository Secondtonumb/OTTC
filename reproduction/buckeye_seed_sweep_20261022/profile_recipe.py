"""Profile full logical batch64 including the longest Buckeye audio, before training."""
import gc
import math
import os
import platform
import time

import numpy as np
import torch
from transformers import set_seed

from reproduction.buckeye_seed_sweep_20261022.common import (
    SSL, SSL_SHA, build_model, Collator, parameter_manifest, sha, write, source_hashes, atomic_weights,
)
from reproduction.timit_divergences_20261022.profile_recipe import gradient_checks
from reproduction.timit_ssl_native10_20261019.validate import verify_ssl_encoder


def attempt(config, data, rows, vocab, processor, output, micro):
    model = optimizer = batch = loss = checked = None
    try:
        set_seed(config['seed'])
        model, topo = build_model(config, vocab, math.ceil(len(data['train']) / 64))
        assert topo is None
        initial = verify_ssl_encoder(model)
        write(output / 'INITIALIZATION_AUDIT.json', dict(encoder=initial, parameters=parameter_manifest(model)))
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        model.cuda().train()
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
        collator = Collator(processor)
        longest = sorted(range(len(rows['train'])), key=lambda index: rows['train'][index]['duration'], reverse=True)
        typical = np.random.default_rng(config['seed']).permutation(len(rows['train'])).tolist()
        tail = len(data['train']) % 64
        cases = [('longest64', longest[:64]), ('representative64', typical[:64])]
        if tail:
            cases.append((f'tail{tail}', typical[:tail]))
        records = []
        for label, indices in cases:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            total = 0.
            for offset in range(0, len(indices), micro):
                batch = collator([data['train'][int(index)] for index in indices[offset:offset + micro]])
                batch = {key: value.cuda() for key, value in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(**batch).loss / len(indices)
                assert torch.isfinite(loss)
                loss.backward()
                total += float(loss.detach())
                batch = loss = None
            gradients = gradient_checks(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            records.append(dict(case=label, logical_batch=len(indices), microbatch=micro, loss=total,
                                max_audio_seconds=max(rows['train'][int(index)]['duration'] for index in indices),
                                seconds=time.monotonic() - begin, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                                gradients=gradients))
        atomic_weights(model, output / 'profile_weights.pt', dict(recipe=config['name'], profile_only=True, epoch=0, step=0))
        checked = torch.load(output / 'profile_weights.pt', map_location='cpu', mmap=True, weights_only=False)
        assert all(value.dtype == torch.float32 for value in checked['model'].values() if value.is_floating_point())
        return records
    finally:
        del checked, model, optimizer, batch, loss
        (output / 'profile_weights.pt').unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()


def run_profile(config, data, rows, vocab, processor, output):
    output.mkdir(parents=True, exist_ok=True)
    assert config['global_batch'] == 64
    assert sha(SSL / 'pytorch_model.bin') == SSL_SHA
    failures = []
    for micro in (64, 32, 16, 8, 4, 2, 1):
        try:
            records = attempt(config, data, rows, vocab, processor, output, micro)
        except torch.cuda.OutOfMemoryError:
            failures.append(dict(microbatch=micro, reason='CUDA out of memory on exact longest64 profile'))
            gc.collect()
            torch.cuda.empty_cache()
            write(output / 'OOM_ATTEMPTS.json', failures)
            continue
        write(output / 'PASS.json', dict(config=config, source_sha256=source_hashes(), microbatch=micro,
              accumulation_steps=64 // micro, batches=records, initial_SSL_verified=True,
              saved_model_dtype='float32', host=platform.node(), job=os.environ.get('PBS_JOBID'),
              oom_attempts=failures, logical_batch=64, distributed_world_size=1))
        return
    raise RuntimeError('No microbatch fits the exact Buckeye profile; training not started')
