"""Profile full logical batch64 including the longest Libri100 audio, before training."""
import gc
import math
import os
import platform
import time

import numpy as np
import torch
from transformers import set_seed

from reproduction.libri100_timit_seed_sweep_20261022.common import (
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
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        dev_indices=sorted(range(len(data['dev'])),key=lambda index:rows['dev'][index]['duration'],reverse=True)[:micro]
        torch.cuda.reset_peak_memory_stats()
        batch=collator([data['dev'][int(index)] for index in dev_indices])
        batch={key:value.cuda() for key,value in batch.items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**batch).loss/len(dev_indices)
        assert torch.isfinite(loss)
        records.append(dict(case='longest_libri_dev_inference',batch=len(dev_indices),
                            max_audio_seconds=max(rows['dev'][int(index)]['duration'] for index in dev_indices),
                            peak_GiB=torch.cuda.max_memory_allocated()/2**30))
        batch=loss=None
        from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
        from reproduction.timit_seed_sweep_20261022.pfer import score_predictions
        target=output/'timit_dev_probe.json'
        probe=evaluate(config,model,rows['timit_dev'][:8],vocab,processor,topo,target,
                       inference_precision='FP32 profile-only model; never a reported trained result')
        assert probe['modes']['forced_alignment']['alignment_failures']==0
        write(output/'PFER_PROBE.json',score_predictions(target.with_suffix('.predictions.jsonl'),probe))
        atomic_weights(model, output / 'profile_weights.pt', dict(recipe=config['name'], profile_only=True, epoch=0, step=0))
        checked = torch.load(output / 'profile_weights.pt', map_location='cpu', mmap=True, weights_only=False)
        assert all(value.dtype == torch.float32 for value in checked['model'].values() if value.is_floating_point())
        if max(record['peak_GiB'] for record in records) >= .9 * torch.cuda.get_device_properties(0).total_memory / 2**30:
            raise MemoryError('Profile exceeds 90% memory safety margin')
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
        print('PROFILE_ATTEMPT',config['name'],'logical_batch=64','microbatch=',micro,flush=True)
        try:
            records = attempt(config, data, rows, vocab, processor, output, micro)
        except (torch.cuda.OutOfMemoryError, MemoryError):
            failures.append(dict(microbatch=micro, reason='CUDA OOM or memory safety margin on exact train/dev profile'))
            gc.collect()
            torch.cuda.empty_cache()
            write(output / 'OOM_ATTEMPTS.json', failures)
            continue
        write(output / 'PASS.json', dict(config=config, source_sha256=source_hashes(), microbatch=micro,
              accumulation_steps=64 // micro, batches=records, initial_SSL_verified=True,
              saved_model_dtype='float32', host=platform.node(), job=os.environ.get('PBS_JOBID'),
              oom_attempts=failures, logical_batch=64, distributed_world_size=1,
              timit_alignment_probe_utterances=8,
              estimated_training_compute_seconds=40 * math.ceil(len(data['train'])/64) * records[1]['seconds'],
              estimate_excludes='epoch dev inference, checkpoint I/O, and final TIMIT evaluation'))
        print('PROFILE_PASS',config['name'],'logical_batch=64','microbatch=',micro,flush=True)
        return
    raise RuntimeError('No microbatch fits the exact Libri100 profile; training not started')
