"""Real dual-view end-to-end GH200 gate; profile models are disposable."""
import gc
import math
import os
import platform
import time
import numpy as np
import torch
from transformers import set_seed
from reproduction.crottc_ag_seed_sweep_20261022.common import build_model,Collator,write,source_hashes,parameter_manifest,SpeechViews,pair_chunks,atomic_weights


def gradient_checks(model):
    groups=dict(last_cnn=list(model.wav2vec2.feature_extractor.conv_layers[-1].parameters()),transformer=list(model.wav2vec2.encoder.parameters()),phone_head=list(model.lm_head.parameters()),
        mass_head=list(model.lm_weight.parameters()))
    report={}
    for name,parameters in groups.items():
        active=[p.grad for p in parameters if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active),name
        norm=sum(float(g.float().square().sum()) for g in active)**.5
        assert norm>0,(name,norm)
        report[name]=norm
    assert all(p.grad is None and not p.requires_grad for layer in model.wav2vec2.feature_extractor.conv_layers[:-1] for p in layer.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return report


def run_profile(config,data,rows,vocab,processor,output):
    from reproduction.crottc_ag_seed_sweep_20261022.train import start_wandb,append
    from reproduction.crottc_ag_seed_sweep_20261022.evaluate import evaluate
    from reproduction.crottc_ag_seed_sweep_20261022.validate import verify_ssl_encoder
    output.mkdir(parents=True,exist_ok=True);set_seed(config['seed']);torch.set_num_threads(4)
    begin=time.monotonic();model,topo=build_model(config,vocab,math.ceil(len(data['train'])/64))
    from reproduction.crottc_ag_seed_sweep_20261022.common import SSL,SSL_SHA,sha
    assert sha(SSL/'pytorch_model.bin')==SSL_SHA
    initial=verify_ssl_encoder(model);initialization_seconds=time.monotonic()-begin
    write(output/'INITIALIZATION_AUDIT.json',dict(original_SSL=initial,parameters=parameter_manifest(model)))
    model.cuda().train();augmenter=SpeechViews('cuda');collator=Collator(processor)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    run=start_wandb(output,config,'profile');records=[];successful=[]
    longest=sorted(range(len(data['train'])),key=lambda i:rows['train'][i]['duration'],reverse=True)
    typical=np.random.default_rng(config['seed']).permutation(len(data['train'])).tolist()
    safety=.85*torch.cuda.get_device_properties(0).total_memory/2**30

    def measure(label,indices,micro):
        start=time.monotonic();base=collator([data['train'][int(i)] for i in indices]);loading=time.monotonic()-start
        torch.cuda.synchronize();start=time.monotonic();base={k:v.cuda(non_blocking=True) for k,v in base.items()}
        torch.cuda.synchronize();transfer=time.monotonic()-start
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        chunks,aug=pair_chunks(model,base,augmenter,micro)
        torch.cuda.synchronize();augmentation=time.monotonic()-start;start=time.monotonic();components={}
        for chunk in chunks:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**chunk).loss/len(indices)
            assert torch.isfinite(loss);loss.backward()
            for key,value in model.last_loss_components.items():components[key]=components.get(key,0.)+float(value)
        grads=gradient_checks(model);torch.cuda.synchronize();compute=time.monotonic()-start
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);start=time.monotonic();optimizer.step();torch.cuda.synchronize()
        opt_seconds=time.monotonic()-start
        record=dict(case=label,**aug,load_seconds=loading,transfer_seconds=transfer,augmentation_seconds=augmentation,
            forward_backward_seconds=compute,optimizer_seconds=opt_seconds,
            total_seconds=loading+transfer+augmentation+compute+opt_seconds,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            gradient_norms=grads,loss_components=components,audio_seconds=sum(rows['train'][int(i)]['duration'] for i in indices))
        record['audio_seconds_per_second']=record['audio_seconds']/record['total_seconds']
        records.append(record);append(output/'batches.jsonl',record)
        run.log({k:v for k,v in record.items() if isinstance(v,(int,float))})
        return record

    for candidate in [16,32,64]:
        try:
            saved=augmenter.speed
            from speechbrain.augment.time_domain import SpeedPerturb
            # SpeedPerturb uses new_freq=orig_freq*speed/100; 105 yields most samples.
            augmenter.speed=SpeedPerturb(orig_freq=16000,speeds=[105],device='cuda')
            worst=measure('longest_logical64',longest[:64],candidate)
            augmenter.speed=saved
            typical_record=measure('representative_logical64',typical[:64],candidate)
            if max(worst['peak_allocated_GiB'],typical_record['peak_allocated_GiB'])<safety:
                successful.append((typical_record['total_seconds'],candidate))
            else:break
        except torch.OutOfMemoryError:
            augmenter.speed=saved;optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            append(output/'batches.jsonl',dict(case='OOM',microbatch=candidate));break
    assert successful,'No safe real-data CROTTC configuration'
    duration,micro=min(successful)
    measure('tail12',typical[:12],micro)
    model.eval();start=time.monotonic()
    checked=evaluate(config,model,rows['dev'][:8],vocab,processor,topo,output/'evaluation_probe.json')
    eval_seconds=time.monotonic()-start
    assert checked['modes']['forced_alignment']['alignment_failures']==0
    start=time.monotonic();atomic_weights(model,output/'profile_weights.pt',dict(recipe=config['name'],epoch=0,step=0,profile_only=True))
    checkpoint_seconds=time.monotonic()-start
    saved=torch.load(output/'profile_weights.pt',map_location='cpu',mmap=True,weights_only=False)
    assert all(v.dtype==torch.float32 for v in saved['model'].values() if v.is_floating_point())
    del saved
    assert all(p.dtype==torch.float32 for p in model.parameters())
    (output/'profile_weights.pt').unlink()
    report=dict(config=config,source_sha256=source_hashes(),microbatch=micro,accumulation_steps=64//micro,
        batches=records,native_stride_verified=model.wave2vec2_config.conv_stride,initial_SSL_verified=True,
        representative_optimizer_step_seconds=duration,eval_probe_seconds=eval_seconds,checkpoint_write_seconds=checkpoint_seconds,
        estimated_40epoch_seconds=40*(73*duration+eval_seconds*50+checkpoint_seconds),
        initialization_seconds=initialization_seconds,host=platform.node(),job=os.environ.get('PBS_JOBID'),
        GPUs=1,DDP=False,model_dtype='float32',saved_weights_dtype='float32',averaging_dtype='float32',model_avg='not instantiated',memory_safety_limit_GiB=safety)
    write(output/'PASS.json',report);run.summary.update({k:v for k,v in report.items() if isinstance(v,(int,float,bool))})
    run.finish();del model,optimizer;gc.collect();torch.cuda.empty_cache()
