"""Native 10/20ms single-process training with existing OTTC cost branches."""
import common
from common import HERE,ROOT,recipe,load_data,build_model,Collator,source_hashes,write,sha,atomic_weights,phones_from_ids,normalize,scorer
import argparse
from contextlib import nullcontext
import gc
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from transformers import Trainer,TrainerCallback,TrainingArguments,set_seed


def microbatch(config):
    if config['corpus']=='timit' and config['frame_ms']==20: return 64
    if config['corpus']=='libri100': return 8 if config['frame_ms']==10 else 16
    return 16 if config['frame_ms']==10 else 32


def start_wandb(output,config,job_type):
    import wandb
    defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    saved=json.loads((output/'wandb_run.json').read_text()) if (output/'wandb_run.json').exists() else {}
    options=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
        id=saved.get('id') or wandb.util.generate_id(),name=config['name']+('-profile' if job_type=='profile' else ''),
        group='native_stride_ottc_20261018',job_type=job_type,config=config,dir=str(output),
        mode=os.environ.get('WANDB_MODE',defaults['mode']),settings=wandb.Settings(init_timeout=45,disable_code=True))
    write(output/'wandb_run.json',dict(id=options['id'],project=options['project'],entity=options['entity'],origin='live',mode=options['mode']))
    if saved and options['mode']=='online': options['resume']='allow'
    try: run=wandb.init(**options)
    except wandb.errors.CommError:
        options['mode']='offline';options.pop('resume',None);run=wandb.init(**options)
    write(output/'wandb_run.json',dict(id=run.id,url=run.url,project=run.project,entity=run.entity,mode=run.settings.mode,origin='live'))
    return run


def append(path,record):
    with path.open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record),flush=True)


def compute_metrics(vocab,topo):
    helpers=scorer()
    def metrics(pred):
        refs=[normalize(phones_from_ids(r,vocab,topo,collapse=False)) for r in pred.label_ids]
        hyps=[normalize(phones_from_ids(r,vocab,topo)) for r in pred.predictions]
        return dict(nosil_PER_percent=helpers['phone_per'](refs,hyps))
    return metrics


def preprocess(logits,labels):
    logits=logits[0] if isinstance(logits,tuple) else logits
    return logits.argmax(-1)


class ChunkedCollator:
    """Keep true 64-example optimizer batches, including an exact short tail."""
    def __init__(self,collator,micro): self.collator,self.micro=collator,micro
    def __call__(self,examples):
        return dict(microbatches=[self.collator(examples[i:i+self.micro]) for i in range(0,len(examples),self.micro)],
                    logical_batch_size=len(examples))


class Policy(TrainerCallback):
    def __init__(self,output,config,rows,vocab,processor,topo,deadline,run):
        self.output,self.config,self.rows,self.vocab,self.processor,self.topo=output,config,rows,vocab,processor,topo
        self.deadline,self.run=deadline,run
        self.best=float('inf')
        if (output/'BEST_DEV.json').exists(): self.best=json.loads((output/'BEST_DEV.json').read_text())['PER']

    def on_step_begin(self,args,state,control,model=None,**kwargs):
        model.step=self.config['step_offset']+state.global_step-1

    def on_step_end(self,args,state,control,**kwargs):
        if time.monotonic()>=self.deadline:
            control.should_training_stop=True;control.should_save=True
        return control

    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        epoch=int(round(state.epoch))
        if abs(state.epoch-epoch)>1e-6: return
        path=self.output/'epoch_weights'/f'epoch-{epoch:03d}.pt'
        metadata=dict(epoch=epoch,step=state.global_step,recipe=self.config['name'],source_sha256=self.config['source_sha256'],
            model_config=model.config.to_dict(),native_conv_stride=self.config['native_conv_stride'])
        if not path.exists():
            atomic_weights(model,path,metadata)
            write(path.with_suffix('.json'),dict(**metadata,weights_sha256=sha(path)))

    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        value=metrics.get('eval_nosil_PER_percent')
        if value is not None and value<self.best:
            self.best=value
            path=self.output/'best_dev.pt'
            atomic_weights(model,path,dict(epoch=state.epoch,step=state.global_step,recipe=self.config['name']))
            write(self.output/'BEST_DEV.json',dict(PER=value,epoch=state.epoch,step=state.global_step,checkpoint=str(path),selection='development greedy no-sil PER only'))
        # Extra audio-only dev boundary metrics; preserve training RNG.
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        training=model.training;model.eval();helpers=scorer();counts=helpers['new_counts']()
        try:
            with torch.inference_mode():
                for offset in range(0,len(self.rows),8):
                    subset=self.rows[offset:offset+8]
                    batch=Collator(self.processor)([dict(wav=r['wav'],labels=r['training_target_ids']) for r in subset])
                    batch.pop('labels');batch={k:v.to('cuda') for k,v in batch.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**batch)
                    for row,path,length in zip(subset,out.logits.argmax(-1).cpu().tolist(),out.hidden_states[1].sum(-1).cpu().tolist()):
                        path=path[:int(length)];counts['utterances']+=1
                        if self.topo:
                            from reproduction.native_stride_ottc_20261018.evaluate import greedy_spans
                            spans=greedy_spans(path,self.vocab,self.topo,self.config['frame_ms']/1000,row['duration'])
                            predicted=[t for _,s,e in spans for t in (s,e)]
                        else:
                            path=np.asarray(path);path[path==41]=0
                            predicted=((np.flatnonzero(path[1:]!=path[:-1])+1)*self.config['frame_ms']/1000).tolist()
                        helpers['score'](counts,row['reference'],predicted,row['duration'])
            summary=helpers['summary'](counts)
            write(self.output/'latest_development_segmentation.json',dict(epoch=state.epoch,step=state.global_step,**summary))
            record=dict(epoch=state.epoch,optimizer_step=state.global_step,**{f'dev/segmentation/{k}':v for k,v in summary['boundaries']['20'].items()})
            append(self.output/'development_segmentation.jsonl',record);self.run.log(record)
        finally:
            model.train(training);random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])

    def on_log(self,args,state,control,logs=None,**kwargs):
        record={**(logs or {}),'optimizer_step':state.global_step,'epoch':state.epoch}
        append(self.output/'history.jsonl',record);self.run.log(record)


class QuietTrainer(Trainer):
    def training_step(self,model,inputs):
        model.train();total=None
        base_step=model.step
        for index,batch in enumerate(inputs['microbatches']):
            model.step=base_step
            # Copy the regfrozen branch once at this optimizer boundary.
            if index and base_step==model.frozen_steps_for_regfrozen-2: model.step+=1
            batch=self._prepare_inputs(batch)
            with self.compute_loss_context_manager(): loss=model(**batch).loss/inputs['logical_batch_size']
            self.accelerator.backward(loss)
            total=loss.detach() if total is None else total+loss.detach()
        return total

    def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None):
        assert len(inputs['microbatches'])==1
        inputs=self._prepare_inputs(inputs['microbatches'][0])
        with torch.no_grad(),self.compute_loss_context_manager():
            out=model(**inputs)
        loss=(out.loss/len(inputs['labels'])).detach()
        if prediction_loss_only:return loss,None,None
        ids=out.logits.argmax(-1).masked_fill(out.hidden_states[1]==0,0)
        return loss,ids.detach(),inputs['labels'].detach()


def profile(config,data,rows,vocab,processor,output):
    output.mkdir(parents=True,exist_ok=True)
    set_seed(config['seed']);batch_size=microbatch(config)
    steps_per_epoch=math.ceil(len(data['train'])/config['global_batch'])
    model,topo=build_model(config,vocab,steps_per_epoch);model.cuda();model.train()
    collator=Collator(processor,topo['mapping'] if topo else None)
    longest_order=sorted(range(len(rows['train'])),key=lambda i:rows['train'][i]['duration'],reverse=True)
    representative_order=np.random.default_rng(42).permutation(len(data['train'])).tolist()
    run=start_wandb(output,config,'profile');records=[]
    # Include real loading, forward, backward, optimizer and frozen-branch phase.
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=.005)
    def measure(name,indices,phase):
        model.step=(config['step_offset']+1 if config['step_offset'] else (model.frozen_steps_for_regfrozen+1 if phase else 0))
        begin=time.monotonic();batch=collator([data['train'][int(i)] for i in indices]);load_seconds=time.monotonic()-begin
        batch={k:v.to('cuda') for k,v in batch.items()};torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(**batch).loss/len(indices)
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
        record=dict(case=name,microbatch=len(indices),raw_loss_per_utterance=float(loss),load_seconds=load_seconds,
            compute_seconds=time.monotonic()-begin,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            audio_seconds=sum(rows['train'][int(i)]['duration'] for i in indices))
        record['audio_seconds_per_second']=record['audio_seconds']/(record['load_seconds']+record['compute_seconds'])
        records.append(record);append(output/'batches.jsonl',record);run.log(record)
        return record
    successful=[]
    for candidate in sorted({batch_size,min(batch_size*2,64),min(batch_size*4,64),64}):
        try:
            longest=measure('longest',longest_order[:candidate],0)
            frozen=measure('frozen_mass_longest',longest_order[:candidate],1)
            representative=measure('representative',representative_order[:candidate],0)
            peak=max(longest['peak_GiB'],frozen['peak_GiB'])
            if peak<90:successful.append((representative['audio_seconds_per_second'],candidate))
            else:break
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            append(output/'batches.jsonl',dict(case='OOM',microbatch=candidate));break
    assert successful,'No safe profiled microbatch'
    batch_size=max(successful)[1]
    config['data_counts']={s:len(r) for s,r in rows.items()}
    write(output/'PASS.json',dict(config=config,source_sha256=source_hashes(),microbatch=batch_size,
        accumulation_steps=config['global_batch']//batch_size,batches=records,native_stride_verified=model.wave2vec2_config.conv_stride,
        host=platform.node(),job=os.environ.get('PBS_JOBID')))
    artifact=__import__('wandb').Artifact(config['name']+'-profile',type='profile');artifact.add_file(str(output/'PASS.json'));run.log_artifact(artifact);run.finish()
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);parser.add_argument('--profile',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=79200);parser.add_argument('--max-steps',type=int,default=0)
    args=parser.parse_args();assert torch.cuda.is_available();torch.set_num_threads(4)
    config=recipe(args.recipe);data,rows,vocab,processor,audit=load_data(config);config['data']=audit
    if args.profile:
        profile(config,data,rows,vocab,processor,HERE/'profiles'/args.recipe);return
    gate=json.loads((HERE/'profiles'/args.recipe/'PASS.json').read_text());assert gate['source_sha256']==source_hashes()
    output=HERE/'models'/args.recipe;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    set_seed(42);micro=gate['microbatch'];accumulation=gate['accumulation_steps']
    steps_per_epoch=math.ceil(len(data['train'])/config['global_batch'])
    model,topo=build_model(config,vocab,steps_per_epoch)
    config.update(microbatch=micro,accumulation_steps=accumulation,planned_steps_per_epoch=steps_per_epoch)
    contract=output/'run_config.json'
    if contract.exists(): assert json.loads(contract.read_text())==config,'Active recipe changed'
    else: write(contract,config)
    processor.save_pretrained(output/'processor');write(output/'vocab_used.json',vocab)
    run=start_wandb(output,config,'training')
    training_args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=config['global_batch'],per_device_eval_batch_size=micro,
        gradient_accumulation_steps=1,num_train_epochs=config['epochs'],learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],weight_decay=.005,bf16=True,fp16=False,seed=42,data_seed=42,
        evaluation_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=False,
        logging_steps=10,report_to=[],dataloader_num_workers=4,dataloader_pin_memory=True,
        remove_unused_columns=False,group_by_length=True,length_column_name='length',max_grad_norm=1.,
        max_steps=args.max_steps if args.max_steps else -1,disable_tqdm=True,save_safetensors=True)
    policy=Policy(output,config,rows['dev'],vocab,processor,topo,time.monotonic()+args.max_seconds,run)
    trainer=QuietTrainer(model=model,args=training_args,train_dataset=data['train'],eval_dataset=data['dev'],
        data_collator=ChunkedCollator(Collator(processor,topo['mapping'] if topo else None),micro),tokenizer=processor,
        compute_metrics=compute_metrics(vocab,topo),callbacks=[policy])
    checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[1]))
    resume=str(checkpoints[-1]) if checkpoints else None
    if not (output/'TRAIN_COMPLETE.json').exists():
        trainer.train(resume_from_checkpoint=resume,ignore_keys_for_eval=['hidden_states','attentions'])
    else:
        from safetensors.torch import load_file
        assert resume
        model.load_state_dict(load_file(str(Path(resume)/'model.safetensors')))
        model.cuda()
        from transformers.trainer_callback import TrainerState
        trainer.state=TrainerState.load_from_json(str(Path(resume)/'trainer_state.json'))
    state=trainer.state;write(output/'LATEST_STATE.json',dict(epoch=state.epoch,step=state.global_step,checkpoint=str(max(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[1])))))
    complete=state.epoch>=config['epochs']-1e-5
    if complete:
        write(output/'TRAIN_COMPLETE.json',dict(epoch=state.epoch,step=state.global_step,recipe=args.recipe))
        from reproduction.native_stride_ottc_20261018.evaluate import final_benchmarks
        final_benchmarks(config,model,rows,vocab,processor,topo,output,run)
    else: write(output/'PAUSED.json',dict(epoch=state.epoch,step=state.global_step,reason='allocation budget; resume same recipe and saved W&B ID'))
    artifact=__import__('wandb').Artifact(args.recipe+'-summaries',type='evaluation')
    for path in output.glob('*.json'): artifact.add_file(str(path),name=path.name)
    run.log_artifact(artifact);run.summary['training_complete']=complete;run.finish()


if __name__=='__main__': main()
