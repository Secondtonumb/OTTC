"""Exact fixed-batch64 gate matching the historical model and precision."""
import gc
import os
import platform
import time

import numpy as np
import torch
from transformers import set_seed

from reproduction.timit_seed_sweep_20261022.common import (
    SSL, SSL_SHA, build_model, Collator, parameter_manifest, sha, write, source_hashes, atomic_weights,
)
from reproduction.timit_divergences_20261022.profile_recipe import gradient_checks
from reproduction.timit_ssl_native10_20261019.validate import verify_ssl_encoder


def run_profile(config, data, rows, vocab, processor, output):
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    assert sha(SSL / 'pytorch_model.bin') == SSL_SHA
    model, topo = build_model(config, vocab, 73)
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
    records = []
    for label, indices in [('longest64', longest[:64]), ('representative64', typical[:64]), ('tail12', typical[:12])]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        batch = collator([data['train'][int(index)] for index in indices])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**{key: value.cuda() for key, value in batch.items()}).loss / len(indices)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = gradient_checks(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        records.append(dict(case=label, batch=len(indices), loss=float(loss), seconds=time.monotonic() - begin,
                            peak_GiB=torch.cuda.max_memory_allocated() / 2**30, gradients=gradients))
    atomic_weights(model, output / 'profile_weights.pt', dict(recipe=config['name'], profile_only=True, epoch=0, step=0))
    checked = torch.load(output / 'profile_weights.pt', map_location='cpu', mmap=True, weights_only=False)
    assert all(value.dtype == torch.float32 for value in checked['model'].values() if value.is_floating_point())
    write(output / 'PASS.json', dict(config=config, source_sha256=source_hashes(), microbatch=64,
          accumulation_steps=1, batches=records, initial_SSL_verified=True, saved_model_dtype='float32',
          host=platform.node(), job=os.environ.get('PBS_JOBID')))
    del checked, model, optimizer
    (output / 'profile_weights.pt').unlink()
    gc.collect()
    torch.cuda.empty_cache()
