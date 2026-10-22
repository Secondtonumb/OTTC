"""Interactive TIMIT divergence training; evaluate every five epochs."""
from reproduction.timit_divergences_20261022 import common
from reproduction.timit_divergences_20261022.common import HERE,ROOT,recipe,load_data,build_model,Collator,source_hashes,write,sha,atomic_weights,phones_from_ids,normalize,scorer,parameter_manifest
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
import signal
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
        group='timit_divergences_20261022',job_type=job_type,config={k:v for k,v in config.items() if k not in ('source_sha256','parameter_manifest')},dir=str(output),
        mode='disabled' if job_type=='profile' else os.environ.get('WANDB_MODE',defaults['mode']),settings=wandb.Settings(init_timeout=45,disable_code=True))
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
        self.stop_requested=False
        self.best=json.loads((output/'BEST_DEV.json').read_text())['PER'] if (output/'BEST_DEV.json').exists() else float('inf')

    def on_step_end(self,args,state,control,**kwargs):
        if self.stop_requested or time.monotonic()>=self.deadline:
            control.should_training_stop=True;control.should_save=True
        return control

    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        epoch=int(round(state.epoch))
        # Trainer also calls this hook when interrupted partway through an epoch.
        if abs(state.epoch-epoch)>1e-6:
            return control
        scheduled=epoch % self.config['eval_every']==0 or epoch==self.config['epochs']
        if scheduled or epoch in self.config['average_epochs']:
            path=self.output/'epoch_weights'/f'epoch-{epoch:03d}.pt'
            if not path.exists():
                atomic_weights(model,path,dict(epoch=epoch,step=state.global_step,
                    recipe=self.config['name'],model_config=model.config.to_dict(),
                    config_sha256=__import__('hashlib').sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest(),
                    source_sha256=self.config['source_sha256']))
        if scheduled:
            # One scoring pass, rather than HF greedy eval plus boundary eval.
            self.on_evaluate(args,state,control,model=model)
            control.should_save=True
        return control

    def on_evaluate(self,args,state,control,model=None,metrics=None,**kwargs):
        # Trainer's greedy diagnostic is retained. Topology recognition and
        # boundary selection use the actual K2 graph on the development set.
        saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        training=model.training
        try:
            from reproduction.timit_divergences_20261022.evaluate import evaluate
            target=self.output/'development'/f'epoch-{int(round(state.epoch)):03d}.json'
            summary=evaluate(self.config,model,self.rows,self.vocab,self.processor,self.topo,target,
                inference_precision='Trainer BF16 autocast; final benchmark reconstructed independently in FP32')
            mode='fsa_text_free' if self.topo else 'text_free'
            value=summary['modes'][mode]['nosil_PER_percent']
            index_path = self.output/'BEST_THREE.json'
            ranked = json.loads(index_path.read_text())['checkpoints'] if index_path.exists() else []
            step = state.global_step
            if not any(item['step'] == step for item in ranked):
                candidate = dict(PER=value, epoch=state.epoch, step=step,
                    path=str(self.output/'best_weights'/f'step-{step:06d}.pt'))
                keep = sorted(ranked + [candidate], key=lambda item:(item['PER'], item['step']))[:3]
                if candidate in keep:
                    atomic_weights(model,Path(candidate['path']),dict(epoch=state.epoch,step=step,
                        recipe=self.config['name'],dev_PER=value,model_config=model.config.to_dict(),
                        config_sha256=__import__('hashlib').sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest(),
                        source_sha256=self.config['source_sha256']))
                    write(index_path,dict(criterion='lowest development PER; test unused',checkpoints=keep))
                    # Only remove weights previously indexed by this new run.
                    for old in ranked:
                        if old not in keep:
                            path=Path(old['path'])
                            assert path.parent == self.output/'best_weights'
                            path.unlink()
                self.best=min(self.best,value)
                write(self.output/'BEST_DEV.json',dict(PER=min(ranked+[candidate],key=lambda x:x['PER'])['PER'],
                    selection='best three by development PER; test unused'))
            record=dict(epoch=state.epoch,optimizer_step=state.global_step,**{'dev/raw_loss_per_utterance':summary['raw_loss_per_utterance']})
            for name,m in summary['modes'].items():
                if 'nosil_PER_percent' in m:record[f'dev/{name}/PER']=m['nosil_PER_percent']
                for ms,bounds in m['boundaries'].items():
                    for key,value in bounds.items():record[f'dev/{name}/{key}_{ms}ms']=value
                if 'boundary_MAE_ms' in m:record[f'dev/{name}/boundary_MAE_ms']=m['boundary_MAE_ms']
            append(self.output/'development_metrics.jsonl',record);self.run.log(record)
        finally:
            model.train(training);random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2]);torch.cuda.set_rng_state_all(saved[3])

    def on_log(self,args,state,control,logs=None,**kwargs):
        record={**(logs or {}),'optimizer_step':state.global_step,'epoch':state.epoch}
        append(self.output/'history.jsonl',record);self.run.log(record)


class QuietTrainer(Trainer):
    def _load_rng_state(self,checkpoint):
        # This recipe resumes only its own trusted local Trainer checkpoints.
        # Old HF calls torch.load without weights_only; PyTorch 2.6's new
        # default rejects the NumPy RNG tuple. Keep the override local here.
        if checkpoint is None:
            return
        saved=torch.load(Path(checkpoint)/'rng_state.pth',map_location='cpu',weights_only=False)
        random.setstate(saved['python'])
        np.random.set_state(saved['numpy'])
        torch.random.set_rng_state(saved['cpu'])
        if torch.cuda.is_available() and 'cuda' in saved:
            if isinstance(saved['cuda'],list):
                torch.cuda.random.set_rng_state_all(saved['cuda'])
            else:
                torch.cuda.random.set_rng_state(saved['cuda'])

    def training_step(self,model,inputs):
        model.train();total=None
        for batch in inputs['microbatches']:
            batch=self._prepare_inputs(batch)
            with self.compute_loss_context_manager():loss=model(**batch).loss/inputs['logical_batch_size']
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
    from reproduction.timit_divergences_20261022.profile_recipe import run_profile
    return run_profile(config,data,rows,vocab,processor,output)


def main():
    parser=argparse.ArgumentParser(description='Interactive TIMIT native10ms OTTC divergence training')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=64,help='utterances per optimizer update; automatically microbatched')
    parser.add_argument('--cost',choices=common.COSTS,required=True)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--dry-run',action='store_true',help='read-only config/data preflight; no GPU, training or W&B')
    parser.add_argument('--profile',action='store_true',help='profile only; no full training')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true',help='resume an existing matching output directory')
    parser.add_argument('--max-seconds',type=int,default=0,help='optional time budget; 0 disables stopping')
    args=parser.parse_args()
    if args.epochs<3 or args.batch_size<1 or args.max_seconds<0 or args.eval_every<1:
        parser.error('epochs>=3, batch-size/eval-every>0, max-seconds>=0 required')
    config=recipe(args.cost,args.epochs,args.batch_size,args.eval_every)
    output=(args.output_dir or HERE/'training'/config['name']).resolve()
    if args.dry_run:
        print(json.dumps(dict(config=config,inputs=common.inspect_inputs(),output_dir=str(output),
            training_started=False,gpu_verified=False),indent=2))
        return
    if not torch.cuda.is_available():
        parser.error('A GPU is required; run inside your interactive GPU allocation')
    if torch.cuda.device_count()!=1:
        parser.error('This recipe uses one GPU; set CUDA_VISIBLE_DEVICES to one device')
    if not torch.cuda.is_bf16_supported():
        parser.error('This recipe requires BF16 support')
    torch.set_num_threads(4)
    data,rows,vocab,processor,audit=load_data(config);config['data']=audit
    profile_only_files={'profile','source_snapshot','.training.lock'}
    if output.exists() and any(p.name not in profile_only_files for p in output.iterdir()) and not args.resume:
        parser.error('output-dir is nonempty; use --resume for the same training contract')
    if args.resume and not (output/'run_config.json').exists():
        parser.error('--resume requires an existing run_config.json')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Reject changed objectives/sources/data before any new profile or optimizer.
    contract=output/'run_config.json'
    if contract.exists():
        prior=json.loads(contract.read_text())
        expected={k:v for k,v in prior.items() if k not in ('microbatch','accumulation_steps','planned_steps_per_epoch')}
        assert expected==config,'Resume configuration/source/data changed'
    common.snapshot_sources(output,config['source_sha256'])
    if args.profile:
        profile(config,data,rows,vocab,processor,output/'profile');return
    gate_path=output/'profile'/'PASS.json'
    if not gate_path.exists():
        profile(config,data,rows,vocab,processor,output/'profile')
    gate=json.loads(gate_path.read_text())
    assert gate['source_sha256']==source_hashes(), 'Profile sources changed'
    assert gate['config']==config, 'Profile configuration changed'
    set_seed(42);micro=gate['microbatch'];accumulation=gate['accumulation_steps']
    steps_per_epoch=math.ceil(len(data['train'])/config['global_batch'])
    model,topo=build_model(config,vocab,steps_per_epoch)
    config.update(microbatch=micro,accumulation_steps=accumulation,planned_steps_per_epoch=steps_per_epoch)
    if contract.exists(): assert json.loads(contract.read_text())==config,'Active recipe changed'
    else: write(contract,config)
    processor.save_pretrained(output/'processor');write(output/'vocab_used.json',vocab)
    write(output/'TRAINABLE_PARAMETERS.json',parameter_manifest(model))
    run=start_wandb(output,config,'training')
    training_args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=config['global_batch'],per_device_eval_batch_size=micro,
        gradient_accumulation_steps=1,num_train_epochs=config['epochs'],learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],weight_decay=.005,bf16=True,fp16=False,seed=42,data_seed=42,
        evaluation_strategy='no',save_strategy='no',save_total_limit=1,load_best_model_at_end=False,
        logging_steps=10,report_to=[],dataloader_num_workers=4,dataloader_pin_memory=True,
        remove_unused_columns=False,group_by_length=True,length_column_name='length',max_grad_norm=1.,
        max_steps=-1,disable_tqdm=True,save_safetensors=True)
    policy=Policy(output,config,rows['dev'],vocab,processor,topo,time.monotonic()+args.max_seconds if args.max_seconds else float('inf'),run)
    def request_stop(signum,frame):
        policy.stop_requested=True
        print('Stop requested; saving full state after the current optimizer step.',flush=True)
    signal.signal(signal.SIGINT,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
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
        write(output/'TRAIN_COMPLETE.json',dict(epoch=state.epoch,step=state.global_step,recipe=config['name']))
        from reproduction.timit_divergences_20261022.evaluate import final_benchmarks
        # Discard Accelerator's BF16 forward wrapper before any final FP32 metric.
        del trainer,model;gc.collect();torch.cuda.empty_cache()
        with torch.inference_mode(False):model,topo=build_model(config,vocab,steps_per_epoch)
        assert not hasattr(model,'_original_forward')
        assert all(p.dtype==torch.float32 for p in model.parameters())
        final_benchmarks(config,model,rows,vocab,processor,topo,output,run)
    else: write(output/'PAUSED.json',dict(epoch=state.epoch,step=state.global_step,reason='allocation budget; resume same recipe and saved W&B ID'))
    run.summary['training_complete']=complete
    if complete:
        selection=json.loads((output/'DEV_SELECTION.json').read_text())
        test=json.loads((output/f"benchmark_{selection['selected']}_test.json").read_text())
        local_final_summary=dict(selected=selection['selected'],test_used_for_selection=False,
            modes=test['modes'],raw_loss_per_utterance=test['raw_loss_per_utterance'])
        write(output/'FINAL_SUMMARY.json',local_final_summary)
    run.finish()



if __name__=='__main__': main()
