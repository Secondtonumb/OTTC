"""Exact recipe GPU gate: load, backward, optimizer, logical batch and K2 eval."""
import gc
import json
import math
import os
import platform
import time
import numpy as np
import torch
from transformers import set_seed
from reproduction.timit_divergences_20261022.common import HERE,build_model,Collator,write,source_hashes,parameter_manifest


def gradient_checks(model):
    groups=dict(last_cnn=list(model.wav2vec2.feature_extractor.conv_layers[-1].parameters()),
                phone_head=list(model.lm_head.parameters()),mass_head=list(model.lm_weight.parameters()))
    report={}
    for name,parameters in groups.items():
        assert all(p.requires_grad for p in parameters),name
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),name
        norm=sum(float(p.grad.float().square().sum()) for p in parameters)**.5
        assert norm>0,(name,norm)
        report[name]=norm
    assert all(p.grad is None and not p.requires_grad for layer in model.wav2vec2.feature_extractor.conv_layers[:-1] for p in layer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return report


def run_profile(config,data,rows,vocab,processor,output):
    from reproduction.timit_divergences_20261022.train import start_wandb,append
    from reproduction.timit_divergences_20261022.evaluate import evaluate
    from reproduction.timit_divergences_20261022.common import SSL,SSL_SHA,sha
    def verify_ssl_encoder(model):
        assert sha(SSL/'pytorch_model.bin')==SSL_SHA
        source=torch.load(SSL/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
        for key,value in model.wav2vec2.state_dict().items():
            source_key=key if key in source else 'wav2vec2.'+key
            assert torch.equal(value.detach().cpu(),source[source_key]),key
        return dict(all_equal=True,expected_weights_sha256=SSL_SHA)
    output.mkdir(parents=True,exist_ok=True);set_seed(42)
    assert torch.cuda.is_available();torch.set_num_threads(4)
    model,topo=build_model(config,vocab,math.ceil(len(data['train'])/config['global_batch']))
    initial=verify_ssl_encoder(model)
    write(output/'INITIALIZATION_AUDIT.json',dict(encoder_exactly_matches_original_SSL=initial,parameters=parameter_manifest(model)))
    model.cuda();model.train();collator=Collator(processor,topo['mapping'] if topo else None)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    run=start_wandb(output,config,'profile');records=[];successful=[]
    longest=sorted(range(len(data['train'])),key=lambda i:rows['train'][i]['duration'],reverse=True)
    typical=np.random.default_rng(42).permutation(len(data['train'])).tolist()
    safety=.85*torch.cuda.get_device_properties(0).total_memory/2**30
    def measure(label,indices,micro):
        begin=time.monotonic();chunks=[]
        for offset in range(0,len(indices),micro):chunks.append(collator([data['train'][int(i)] for i in indices[offset:offset+micro]]))
        loading=time.monotonic()-begin;optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        for chunk in chunks:
            batch={k:v.cuda(non_blocking=True) for k,v in chunk.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss/len(indices)
            assert torch.isfinite(loss);loss.backward()
        grads=gradient_checks(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
        compute=time.monotonic()-begin
        record=dict(case=label,microbatch=micro,logical_batch=len(indices),load_seconds=loading,compute_seconds=compute,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30,gradient_norms=grads,raw_last_micro_loss_per_logical_batch=float(loss),
            audio_seconds=sum(rows['train'][int(i)]['duration'] for i in indices))
        record['audio_seconds_per_second']=record['audio_seconds']/(loading+compute)
        records.append(record);append(output/'batches.jsonl',record)
        run.log({k:v for k,v in record.items() if not isinstance(v,dict)})
        return record
    for candidate in sorted(set(min(n,config['global_batch']) for n in [1,2,4,8,16,32,64])):
        try:
            worst=measure('longest_inputs',longest[:candidate],candidate)
            logical=measure('representative_logical_batch',typical[:config['global_batch']],candidate)
            if worst['peak_GiB']<safety and logical['peak_GiB']<safety:
                successful.append((logical['load_seconds']+logical['compute_seconds'],candidate))
            else:break
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            append(output/'batches.jsonl',dict(case='OOM',microbatch=candidate));break
    assert successful,'No safe real-data profile configuration'
    duration,micro=min(successful)
    # Exercise real dense K2 search and supplied-phone FA before production.
    model.eval();begin=time.monotonic()
    checked=evaluate(config,model,rows['dev'][:8],vocab,processor,topo,output/'evaluation_probe.json',inference_precision='FP32 profile model after BF16 optimizer probe')
    eval_seconds=time.monotonic()-begin
    assert checked['modes']['forced_alignment']['alignment_failures']==0
    if topo:assert checked['modes']['fsa_text_free']['alignment_failures']==0
    # Forecast compute/evaluation time; checkpoint I/O is not included.
    checkpoint_seconds=0.0
    evaluations=math.ceil(config['epochs']/config['eval_every'])
    estimate=(config['epochs']*math.ceil(len(data['train'])/config['global_batch'])*duration
              +evaluations*eval_seconds*len(rows['dev'])/8)
    report=dict(config=config,source_sha256=source_hashes(),microbatch=micro,accumulation_steps=math.ceil(config['global_batch']/micro),
        batches=records,native_stride_verified=model.wave2vec2_config.conv_stride,initial_SSL_verified=True,
        frozen_CNN_layers=[0,1,2,3,4,5],trainable_CNN_layers=[6],all_heads_trainable=True,
        representative_optimizer_step_seconds=duration,eval_probe_seconds=eval_seconds,checkpoint_write_seconds=checkpoint_seconds,
        estimated_training_seconds=estimate,scheduled_dev_evaluations=evaluations,
        estimate_excludes='checkpoint I/O and final FP32 dev/test comparisons',
        full_resume_state_required=True,output_beam=8,
        host=platform.node(),job=os.environ.get('PBS_JOBID'),memory_safety_limit_GiB=safety)
    write(output/'PASS.json',report)
    run.summary.update({k:v for k,v in report.items() if isinstance(v,(int,float,bool))})
    run.finish();del model,optimizer;gc.collect();torch.cuda.empty_cache()
