"""Bounded real-data batch64/128/256 profiling for original OT-weighted CE."""
import bootstrap
from bootstrap import ROOT,HERE,source_hashes
import gc,json,os,platform,time
import torch
from transformers import set_seed
import run_exp
from model import WavLMForOTTC,WavLMOTTCConfig
from recipe import settings


def candidate(batch_size,data,vocab,processor,base_config):
    set_seed(42)
    recipe=settings(batch_size)
    output=HERE/'profile'/f'batch{batch_size}';output.mkdir(parents=True,exist_ok=True)
    config={**base_config,'batch_size':batch_size,'bins_metric':'crossentropy',
        'suffix':f'wavlm_ce_profile_b{batch_size}_20261013',
        'frozen_steps_for_regfrozen':recipe['frozen_steps_for_regfrozen'],
        'warmup_steps':recipe['warmup_steps']}
    (output/'dataset_sizes.json').write_text(json.dumps({k:len(v) for k,v in data.items()}))
    c=WavLMOTTCConfig(wav2vec2_path=str(ROOT/'reproduction/models/wavlm-large'),vocab_size=len(vocab),
        wav2vec2_output_dropout=.1,final_dropout=.1,bins_metric='crossentropy',final_layers='linear',
        forward_type='regfrozen',frozen_steps_for_regfrozen=recipe['frozen_steps_for_regfrozen'])
    begin=time.monotonic();model=WavLMForOTTC(c);model.freeze_feature_encoder();model.cuda()
    assert model.jsd is False and model.wav2vec2.config.model_type=='wavlm'
    assert model.wav2vec2.encoder.gradient_checkpointing
    init_seconds=time.monotonic()-begin
    trainer=run_exp.create_trainer(model,data,processor,config,lambda _: {},output)
    trainer.args.warmup_steps=recipe['warmup_steps']
    trainer.create_optimizer_and_scheduler(num_training_steps=recipe['total_steps'])
    model,optimizer=trainer.accelerator.prepare(model,trainer.optimizer)
    loader=trainer.get_train_dataloader();iterator=None;rows=[]
    import wandb
    try:
        for phase in ['before_alignment_freeze','after_alignment_freeze','longest_batch']:
            iterator=iter(loader) if phase!='longest_batch' else None
            raw=trainer.accelerator.unwrap_model(model)
            raw.step=recipe['frozen_steps_for_regfrozen']-2 if phase=='after_alignment_freeze' else 0
            for index in range(len(loader) if phase!='longest_batch' else 8):
                torch.cuda.synchronize();begin=time.monotonic()
                if phase=='longest_batch':
                    lengths=data['train']['length']
                    selected=sorted(range(len(lengths)),key=lengths.__getitem__,reverse=True)[:batch_size]
                    batch=trainer.data_collator([data['train'][i] for i in selected])
                else:batch=next(iterator)
                loading=time.monotonic()-begin
                torch.cuda.synchronize();begin=time.monotonic();batch=trainer._prepare_inputs(batch);torch.cuda.synchronize()
                transfer=time.monotonic()-begin
                torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);begin=time.monotonic()
                loss=trainer.training_step(model,batch);torch.cuda.synchronize();compute=time.monotonic()-begin
                assert torch.isfinite(loss),loss
                begin=time.monotonic()
                scale_before=trainer.accelerator.scaler.get_scale()
                norm=trainer.accelerator.clip_grad_norm_(model.parameters(),trainer.args.max_grad_norm)
                finite=bool(torch.isfinite(norm))
                optimizer.step()
                skipped=trainer.accelerator.optimizer_step_was_skipped
                if not skipped:trainer.lr_scheduler.step()
                if not finite and not skipped:raise RuntimeError('Nonfinite gradients were not protected by GradScaler')
                torch.cuda.synchronize();update=time.monotonic()-begin
                record=dict(batch_size=batch_size,phase=phase,index=index,loss=float(loss),
                    grad_norm=float(norm) if finite else None,gradient_finite=finite,amp_step_skipped=skipped,
                    grad_scale_before=scale_before,grad_scale_after=trainer.accelerator.scaler.get_scale(),
                    loading_seconds=loading,transfer_seconds=transfer,forward_backward_seconds=compute,
                    optimizer_seconds=update,total_seconds=loading+transfer+compute+update,
                    utterances=len(batch['input_values']),padded_samples=int(batch['input_values'].numel()),
                    max_audio_seconds=batch['input_values'].shape[1]/16000,
                    max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
                rows.append(record);print('CE_PROFILE',json.dumps(record),flush=True)
                (output/'batches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if wandb.run:wandb.run.log(record)
                if phase=='longest_batch' and finite and not skipped:break
            phase_rows=[r for r in rows if r['phase']==phase]
            minimum=1 if phase=='longest_batch' else 3
            if sum(r['gradient_finite'] and not r['amp_step_skipped'] for r in phase_rows)<minimum:
                raise RuntimeError('GradScaler did not produce enough finite CE updates in '+phase)
        # Compare the same complete 4,620-utterance input set for every batch;
        # include cold loading to reflect actual end-to-end epoch throughput.
        typical=[r for r in rows if r['phase']!='longest_batch']
        total_seconds=sum(r['total_seconds'] for r in typical)
        throughput=sum(r['utterances'] for r in typical)/total_seconds
        peak=max(r['max_allocated_GiB'] for r in rows)
        safe=peak<.9*torch.cuda.get_device_properties(0).total_memory/2**30
        result=dict(batch_size=batch_size,safe=safe,utterances_per_second=throughput,
            mean_step_seconds=total_seconds/len(typical),peak_allocated_GiB=peak,
            init_seconds=init_seconds,longest_input_batch_safe=True,
            amp_skipped_steps=sum(r['amp_step_skipped'] for r in rows),
            finite_optimizer_updates=sum(r['gradient_finite'] and not r['amp_step_skipped'] for r in rows),
            recipe=recipe,batches=rows)
        if wandb.run:wandb.run.summary.update({'profile_pass':safe,'utterances_per_second':throughput,'peak_allocated_GiB':peak})
        return result
    except torch.cuda.OutOfMemoryError:
        if wandb.run:wandb.run.summary.update({'profile_pass':False,'error':'CUDA OOM'})
        raise
    finally:
        run_exp.finish_training_run()
        del iterator,loader,trainer,model,optimizer


def main():
    assert torch.cuda.is_available();torch.set_num_threads(4);set_seed(42)
    output=HERE/'profile';output.mkdir(parents=True,exist_ok=True)
    if (output/'PASS.json').exists() and json.loads((output/'PASS.json').read_text())['source_sha256']==source_hashes():
        print('Matching CE batch profile already passed; reuse evidence',flush=True);return
    (output/'PASS.json').unlink(missing_ok=True)
    old=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd'
    config=json.loads((old/'run_config.json').read_text());tracking=run_exp.wandb_defaults()
    config.update(encoder_name='wavlm-large',bins_metric='crossentropy',
        wandb_project=os.environ.get('WANDB_PROJECT',tracking['project']),
        wandb_entity=os.environ.get('WANDB_ENTITY',tracking['entity']),
        wandb_mode=os.environ.get('WANDB_MODE','online'),logging_steps=1,continue_if_exist=False)
    data,vocab,processor=run_exp.load_dataset_and_processor(config,True)
    assert vocab==json.loads((old/'vocab_used.json').read_text())
    assert {k:len(v) for k,v in data.items()}==json.loads((old/'dataset_sizes.json').read_text())
    results={}
    for batch_size in (64,128,256):
        try:results[str(batch_size)]=candidate(batch_size,data,vocab,processor,config)
        except torch.cuda.OutOfMemoryError as error:
            results[str(batch_size)]=dict(batch_size=batch_size,safe=False,error=str(error))
            print('CE_PROFILE_OOM',batch_size,flush=True)
        gc.collect();torch.cuda.empty_cache()
        (output/'candidates.json').write_text(json.dumps(results,indent=2)+'\n')
    safe={k:v for k,v in results.items() if v.get('safe') and int(k)>=128}
    if not safe:raise RuntimeError('Neither batch128 nor256 passed longest-input safety')
    chosen=min(safe,key=lambda k:int(k))
    if '128' in safe and '256' in safe and safe['256']['utterances_per_second']>=1.05*safe['128']['utterances_per_second']:
        chosen='256'
    receipt=dict(job=os.environ.get('PBS_JOBID'),host=platform.node(),gpu=torch.cuda.get_device_name(),
        world_size=1,ddp_seconds=0,safe_batches=list(safe),selected_batch_size=int(chosen),
        selection='Choose256 only with at least5percent more measured throughput than128; otherwise128',
        candidates=results,train_fingerprint=data['train']._fingerprint,
        kaldi_test_fingerprint=data['kaldi_test']._fingerprint,source_sha256=source_hashes())
    (output/'PASS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('CE_SELECTED_BATCH',chosen,flush=True)


if __name__=='__main__':main()
