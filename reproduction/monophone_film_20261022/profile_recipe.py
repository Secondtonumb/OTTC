"""Local-only exact-model, real-TIMIT GPU gate before formal training."""
import gc
import math
import os
import platform
import time

import numpy as np
import torch
from transformers import set_seed

from reproduction.monophone_film_20261022.common import (
    SSL, SSL_SHA, build_model, Collator, parameter_manifest, sha, write, source_hashes, atomic_weights,
)


def gradient_checks(model):
    groups = dict(last_cnn=model.wav2vec2.feature_extractor.conv_layers[-1],
                  ctc=model.ctc_head, ottc=model.lm_head, mass=model.lm_weight,
                  film_final=model.context_film.net[-1])
    report = {}
    for name, module in groups.items():
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        norm = sum(float(g.float().square().sum()) for g in gradients)**.5
        assert norm > 0, (name, norm)
        report[name] = norm
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None and not p.requires_grad
               for layer in model.wav2vec2.feature_extractor.conv_layers[:-1] for p in layer.parameters())
    return report


def run_profile(config, data, rows, vocab, processor, output):
    from reproduction.timit_ssl_native10_20261019.validate import verify_ssl_encoder
    from reproduction.monophone_film_20261022.evaluate import evaluate
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    assert sha(SSL / 'pytorch_model.bin') == SSL_SHA
    model, topo = build_model(config, vocab, math.ceil(len(data['train'])/64))
    initial = verify_ssl_encoder(model)
    write(output/'INITIALIZATION_AUDIT.json', dict(encoder=initial, parameters=parameter_manifest(model)))
    model.cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collator = Collator(processor)
    longest = sorted(range(len(rows['train'])), key=lambda i: rows['train'][i]['duration'], reverse=True)
    typical = np.random.default_rng(config['seed']).permutation(len(rows['train'])).tolist()
    records, successful = [], []
    safety = .85 * torch.cuda.get_device_properties(0).total_memory / 2**30
    def measure(label, indices, micro):
        optimizer.zero_grad(set_to_none=True)
        model.pop_film_loss_statistics()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        for offset in range(0,len(indices),micro):
            batch = collator([data['train'][int(i)] for i in indices[offset:offset+micro]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**{k:v.cuda() for k,v in batch.items()}).loss / len(indices)
            assert torch.isfinite(loss)
            loss.backward()
        gradients = gradient_checks(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        raw = model.pop_film_loss_statistics().cpu().tolist()
        record = dict(case=label, logical_batch=len(indices), microbatch=micro,
                      seconds=time.monotonic()-begin, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                      gradient_norms=gradients, raw_loss_statistics=raw,
                      audio_seconds=sum(rows['train'][int(i)]['duration'] for i in indices))
        record['audio_seconds_per_second'] = record['audio_seconds']/record['seconds']
        records.append(record)
        write(output/'BATCHES.json',records)
        print('PROFILE',record,flush=True)
        return record
    for micro in (16,32,64):
        try:
            worst = measure('longest_logical64',longest[:64],micro)
            representative = measure('representative_logical64',typical[:64],micro)
            if max(worst['peak_GiB'],representative['peak_GiB']) < safety:
                successful.append((representative['seconds'],micro))
            else:
                break
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            model.pop_film_loss_statistics()
            gc.collect();torch.cuda.empty_cache()
            write(output/f'OOM_micro{micro}.json',dict(microbatch=micro,job=os.environ.get('PBS_JOBID')))
            break
    assert successful, 'No safe full-model real-TIMIT configuration'
    seconds,micro = min(successful)
    measure('tail12',typical[:12],micro)
    model.eval()
    checked = evaluate(config,model,rows['dev'][:8],vocab,processor,topo,output/'evaluation_probe.json')
    assert checked['modes']['forced_alignment']['alignment_failures'] == 0
    assert checked['ctc_head']['modes']['forced_alignment']['alignment_failures'] == 0
    path = output/'profile_weights.pt'
    atomic_weights(model,path,dict(recipe=config['name'],epoch=0,step=0,profile_only=True,model_config=model.config.to_dict()))
    saved = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert all(v.dtype == torch.float32 for v in saved['model'].values() if v.is_floating_point())
    del saved
    # Profile output is a new disposable probe, never a production checkpoint.
    path.unlink()
    write(output/'PASS.json',dict(config=config,source_sha256=source_hashes(),microbatch=micro,
        accumulation_steps=64//micro,batches=records,initial_SSL_verified=True,
        saved_model_dtype='float32',averaging_dtype='float32',model_avg='not instantiated',
        representative_optimizer_step_seconds=seconds,host=platform.node(),job=os.environ.get('PBS_JOBID')))
    del model,optimizer
    gc.collect();torch.cuda.empty_cache()
