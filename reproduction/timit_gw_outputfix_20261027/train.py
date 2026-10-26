"""Native 10/20ms single-process training with existing OTTC cost branches."""
from reproduction.timit_gw_outputfix_20261027 import common
from reproduction.timit_gw_outputfix_20261027.common import HERE,ROOT,recipe,load_data,build_model,Collator,source_hashes,write,sha,atomic_weights,phones_from_ids,normalize,scorer,parameter_manifest
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
        id=saved.get('id') or wandb.util.generate_id(),name='SSL-native10 | '+config['name']+('-profile' if job_type=='profile' else ''),
        group='timit_gw_outputfix_20261027',job_type=job_type,config={k:v for k,v in config.items() if k not in ('source_sha256','parameter_manifest')},dir=str(output),
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
        self.best=json.loads((output/'BEST_DEV.json').read_text())['PER'] if (output/'BEST_DEV.json').exists() else float('inf')

    def on_step_end(self,args,state,control,**kwargs):
        if time.monotonic()>=self.deadline:
            control.should_training_stop=True;control.should_save=True
        return control

    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        epoch=int(round(state.epoch))
        if abs(state.epoch-epoch)>1e-6:
            return control
        atomic_weights(model,self.output/'epoch_weights'/f'epoch-{epoch:03d}.pt',
            dict(epoch=epoch,step=state.global_step,recipe=self.config['name'],model_config=model.config.to_dict(),source_sha256=self.config['source_sha256']))
        control.should_save=True
        return control

    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        # Trainer's greedy diagnostic is retained. Topology recognition and
        # boundary selection use the actual K2 graph on the development set.
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        training=model.training
        try:
            from reproduction.timit_gw_outputfix_20261027.evaluate import evaluate
            epoch=int(round(state.epoch))
            filename=(f'epoch-{epoch:03d}.json' if abs(state.epoch-epoch)<1e-6
                      else f'partial-step-{state.global_step:06d}.json')
            target=self.output/'development'/filename
            summary=evaluate(self.config,model,self.rows,self.vocab,self.processor,self.topo,target,
                inference_precision='Trainer BF16 autocast; final benchmark reconstructed independently in FP32')
            mode='fsa_text_free' if self.topo else 'text_free'
            value=summary['modes'][mode]['nosil_PER_percent']
            common.retain_best(self.output,model,value,state.epoch,state.global_step,self.config)
            self.best=min(self.best,value)
            record=dict(epoch=state.epoch,optimizer_step=state.global_step,**{'dev/raw_loss_per_utterance':summary['raw_loss_per_utterance']})
            for name,m in summary['modes'].items():
                if 'nosil_PER_percent' in m:record[f'dev/{name}/PER']=m['nosil_PER_percent']
                for ms,bounds in m['boundaries'].items():
                    for key,value in bounds.items():record[f'dev/{name}/{key}_{ms}ms']=value
                if 'boundary_MAE_ms' in m:record[f'dev/{name}/boundary_MAE_ms']=m['boundary_MAE_ms']
            append(self.output/'development_metrics.jsonl',record)  # evaluation stays local
        finally:
            model.train(training);random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])

    def on_log(self,args,state,control,logs=None,model=None,**kwargs):
        logs={k:v for k,v in (logs or {}).items() if not k.startswith('eval_')}
        stats=model.pop_gw_statistics() if model is not None else None
        if stats is not None:
            count=float(stats.pop('examples'))
            logs.update({f'gw/{k}':float(v)/count for k,v in stats.items()})
        record={**(logs or {}),'optimizer_step':state.global_step,'epoch':state.epoch}
        append(self.output/'history.jsonl',record);self.run.log(record)


class QuietTrainer(Trainer):
    def _load_rng_state(self,checkpoint):
        if checkpoint is None:
            return
        saved=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        torch.random.set_rng_state(saved['cpu'])
        if 'cuda' in saved:
            if isinstance(saved['cuda'],list):
                torch.cuda.random.set_rng_state_all(saved['cuda'])
            else:
                torch.cuda.random.set_rng_state(saved['cuda'])

    def create_scheduler(self,num_training_steps,optimizer=None):
        full_steps=math.ceil(len(self.train_dataset)/self.args.per_device_train_batch_size)*40
        return super().create_scheduler(full_steps,optimizer=optimizer)

    def training_step(self,model,inputs):
        model.train();total=None
        for batch in inputs['microbatches']:
            batch=self._prepare_inputs(batch)
            with self.compute_loss_context_manager():loss=model(**batch).loss/inputs['logical_batch_size']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite GW training loss; do not mark experiment complete')
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
    from reproduction.timit_gw_outputfix_20261027.profile_recipe import run_profile
    return run_profile(config,data,rows,vocab,processor,output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);parser.add_argument('--profile',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--max-seconds',type=int,default=79200);parser.add_argument('--max-steps',type=int,default=0)
    args=parser.parse_args();assert torch.cuda.is_available();torch.set_num_threads(4)
    config=recipe(args.recipe);data,rows,vocab,processor,audit=load_data(config);config['data']=audit
    assert audit==json.loads((Path(config['baseline'])/'run_config.json').read_text())['data']
    output=HERE/'models'/args.recipe
    common.snapshot_sources(output,config['source_sha256'])
    if args.profile:
        profile(config,data,rows,vocab,processor,HERE/'profiles'/args.recipe);return
    gate=json.loads((HERE/'profiles'/args.recipe/'PASS.json').read_text());assert gate['source_sha256']==source_hashes()
    output=HERE/'models'/args.recipe;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert gate['config']==config,'Profile config/data changed'
    set_seed(config['seed']);micro=gate['microbatch'];accumulation=gate['accumulation_steps']
    steps_per_epoch=math.ceil(len(data['train'])/config['global_batch'])
    model,topo=build_model(config,vocab,steps_per_epoch)
    config.update(microbatch=micro,accumulation_steps=accumulation,planned_steps_per_epoch=steps_per_epoch)
    contract=output/'run_config.json'
    if contract.exists(): assert json.loads(contract.read_text())==config,'Active recipe changed'
    else: write(contract,config)
    processor.save_pretrained(output/'processor');write(output/'vocab_used.json',vocab)
    write(output/'TRAINABLE_PARAMETERS.json',parameter_manifest(model))
    run=start_wandb(output,config,'training')
    training_args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=config['global_batch'],per_device_eval_batch_size=micro,
        gradient_accumulation_steps=1,num_train_epochs=config['epochs'],learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],weight_decay=.005,bf16=True,fp16=False,seed=config['seed'],data_seed=config['seed'],
        evaluation_strategy='epoch',save_strategy='no',save_total_limit=1,load_best_model_at_end=False,
        logging_steps=10,report_to=[],dataloader_num_workers=4,dataloader_pin_memory=True,
        remove_unused_columns=False,group_by_length=True,length_column_name='length',max_grad_norm=1.,
        max_steps=(steps_per_epoch*config['pilot_epochs'] if args.pilot else args.max_steps) or -1,disable_tqdm=True,save_safetensors=True)
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
    trainer._save_checkpoint(model,None)
    state=trainer.state;write(output/'LATEST_STATE.json',dict(epoch=state.epoch,step=state.global_step,checkpoint=str(max(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[1])))))
    complete=state.epoch>=config['epochs']-1e-5
    if complete:
        write(output/'TRAIN_COMPLETE.json',dict(epoch=state.epoch,step=state.global_step,recipe=args.recipe))
        from reproduction.timit_gw_outputfix_20261027.evaluate import final_benchmarks
        # Discard Accelerator's BF16 forward wrapper before any final FP32 metric.
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        with torch.inference_mode(False):model,topo=build_model(config,vocab,steps_per_epoch)
        assert not hasattr(model,'_original_forward')
        assert all(p.dtype==torch.float32 for p in model.parameters())
        final_benchmarks(config,model,rows,vocab,processor,topo,output,run)
    else:
        pilot_done=args.pilot and state.epoch>=config['pilot_epochs']-1e-5
        write(output/'PAUSED.json',dict(epoch=state.epoch,step=state.global_step,
            reason='pilot finished; full 40 epoch training incomplete' if pilot_done else 'allocation budget; resumable'))
        if pilot_done:
            development=output/'development'/f"epoch-{config['pilot_epochs']:03d}.json"
            assert development.exists()
            write(output/'PILOT_COMPLETE.json',dict(epoch=state.epoch,step=state.global_step,
                development=str(development),test_evaluated=False,full_training_complete=False))
    run.summary['training_complete']=complete
    run.finish()



if __name__=='__main__': main()
