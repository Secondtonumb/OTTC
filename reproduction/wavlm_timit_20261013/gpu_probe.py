"""Exact-recipe real-batch profiling and a conservative training gate."""
import bootstrap
from bootstrap import ROOT,HERE,source_hashes
import hashlib,json,os,platform,time
import torch
from transformers import set_seed
import run_exp
from model import WavLMForOTTC,WavLMOTTCConfig


def main():
    completed=ROOT/'reproduction/results/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013/TRAIN_COMPLETE.json'
    if completed.exists():
        print('Production already trained; no additional profile needed',flush=True);return
    assert torch.cuda.is_available()
    torch.set_num_threads(4);set_seed(42)
    output=HERE/'profile';output.mkdir(parents=True,exist_ok=True)
    (output/'PASS.json').unlink(missing_ok=True)
    old=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd'
    config=json.loads((old/'run_config.json').read_text())
    tracking=run_exp.wandb_defaults()
    config.update(encoder_name='wavlm-large',suffix='wavlm_timit_profile_20261013',
        wandb_project=os.environ.get('WANDB_PROJECT',tracking['project']),
        wandb_entity=os.environ.get('WANDB_ENTITY',tracking['entity']),
        wandb_mode=os.environ.get('WANDB_MODE','online'),logging_steps=1,continue_if_exist=False)
    begin=time.monotonic()
    data,vocab,processor=run_exp.load_dataset_and_processor(config,True)
    assert vocab==json.loads((old/'vocab_used.json').read_text())
    assert {k:len(v) for k,v in data.items()}==json.loads((old/'dataset_sizes.json').read_text())
    (output/'dataset_sizes.json').write_text(json.dumps({k:len(v) for k,v in data.items()}))
    data_seconds=time.monotonic()-begin
    # Checkpointing-free construction is profiled separately from update time.
    c=WavLMOTTCConfig(wav2vec2_path=str(ROOT/'reproduction/models/wavlm-large'),
        vocab_size=len(vocab),wav2vec2_output_dropout=.1,final_dropout=.1,bins_metric='jsd',
        final_layers='linear',forward_type='regfrozen',frozen_steps_for_regfrozen=2165)
    begin=time.monotonic();model=WavLMForOTTC(c);model.freeze_feature_encoder();model.cuda()
    init_seconds=time.monotonic()-begin
    assert model.wav2vec2.config.model_type=='wavlm' and model.wav2vec2.config.conv_stride==[5,2,2,2,2,2,2]
    assert model.wav2vec2.encoder.gradient_checkpointing
    official=torch.load(ROOT/'reproduction/models/wavlm-large/pytorch_model.bin',map_location='cpu',weights_only=False)
    for key in ['feature_projection.projection.weight','encoder.layers.0.attention.q_proj.weight',
                'encoder.layers.23.attention.q_proj.weight']:
        assert torch.equal(model.wav2vec2.state_dict()[key].cpu(),official[key]),key
    for old_name,new_name in [('weight_g','parametrizations.weight.original0'),('weight_v','parametrizations.weight.original1')]:
        stem='encoder.pos_conv_embed.conv.'
        assert torch.equal(model.wav2vec2.state_dict()[stem+new_name].cpu(),official[stem+old_name])
    del official
    trainer=run_exp.create_trainer(model,data,processor,config,lambda _: {},output)
    trainer.create_optimizer_and_scheduler(num_training_steps=2920)
    model,optimizer=trainer.accelerator.prepare(model,trainer.optimizer)
    loader=trainer.get_train_dataloader();iterator=iter(loader)
    rows=[]
    import wandb
    try:
        for phase in ['before_alignment_freeze','after_alignment_freeze','longest_batch64']:
            raw=trainer.accelerator.unwrap_model(model)
            raw.step=2163 if phase=='after_alignment_freeze' else 0
            for index in range(3 if phase!='longest_batch64' else 1):
                torch.cuda.synchronize();start=time.monotonic()
                if phase=='longest_batch64':
                    lengths=data['train']['length']
                    indices=sorted(range(len(lengths)),key=lengths.__getitem__,reverse=True)[:64]
                    batch=trainer.data_collator([data['train'][i] for i in indices])
                else:batch=next(iterator)
                loading=time.monotonic()-start
                torch.cuda.synchronize();start=time.monotonic();batch=trainer._prepare_inputs(batch);torch.cuda.synchronize()
                transfer=time.monotonic()-start
                torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True);start=time.monotonic()
                loss=trainer.training_step(model,batch)
                torch.cuda.synchronize();compute=time.monotonic()-start;assert torch.isfinite(loss),loss
                start=time.monotonic();trainer.accelerator.clip_grad_norm_(model.parameters(),trainer.args.max_grad_norm)
                optimizer.step();trainer.lr_scheduler.step();torch.cuda.synchronize()
                update=time.monotonic()-start
                record=dict(phase=phase,index=index,loss=float(loss),loading_seconds=loading,
                    transfer_seconds=transfer,forward_backward_seconds=compute,optimizer_seconds=update,
                    total_seconds=loading+transfer+compute+update,batch_size=len(batch['input_values']),
                    padded_samples=int(batch['input_values'].numel()),max_audio_seconds=batch['input_values'].shape[1]/16000,
                    max_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    max_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
                rows.append(record);print(json.dumps(record),flush=True)
                (output/'batches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if wandb.run:wandb.run.log(record)
        typical=[r['total_seconds'] for r in rows if r['phase']!='longest_batch64' and r['index']>0]
        result=dict(job=os.environ.get('PBS_JOBID'),host=platform.node(),gpu=torch.cuda.get_device_name(),
            data_seconds=data_seconds,init_seconds=init_seconds,world_size=1,ddp_seconds=0,
            mean_step_seconds=sum(typical)/len(typical),estimated_train_2920_steps_seconds=2920*sum(typical)/len(typical),
            longest_input_batch_safe=True,pretrained_tensor_samples_identical=True,batches=rows,
            train_fingerprint=data['train']._fingerprint,kaldi_test_fingerprint=data['kaldi_test']._fingerprint,
            source_sha256=source_hashes())
        (output/'PASS.json').write_text(json.dumps(result,indent=2)+'\n')
        if wandb.run:wandb.run.summary.update({'profile_pass':True,'world_size':1,'ddp_seconds':0})
    finally:run_exp.finish_training_run()


if __name__=='__main__':main()
