"""Train/evaluate four-head English CTC and OTTC models with live W&B."""
import argparse
import hashlib
import json
import math
from segmentation_metrics import r_value_percent
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'reproduction/vendor'),str(ROOT)]
import numpy as np
import soundfile as sf
import torch
import transformers
transformers.utils.import_utils._peft_available=False
transformers.utils.import_utils._bitsandbytes_available=False
from torch.utils.data import Dataset,DataLoader
from transformers import Trainer,TrainingArguments,TrainerCallback,Wav2Vec2Config,set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import seed_worker
import jiwer

from multihead import HEADS,MultiPhoneConfig,MultiPhoneModel
from ottc.utils.wandb_tracking import start_training_run,finish_training_run,LossCurveCallback


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def apply_batch_policy(args):
    """Use the measured Libri policy at a new process launch, retaining batch 64."""
    if not args.batch_policy.exists():
        return None
    policy=json.loads(args.batch_policy.read_text())
    if str(args.data.resolve())!=policy['data_path'] or args.objective not in policy['objectives']:
        return None
    if args.effective_batch_size!=policy['effective_batch_size']:
        raise ValueError('Throughput policy requires the recorded effective batch size')
    requested=args.batch_size
    args.batch_size=policy['microbatch']
    if args.batch_size<=0 or args.effective_batch_size%args.batch_size:
        raise ValueError('Invalid policy microbatch')
    applied={**policy,'requested_microbatch':requested,'policy_sha256':file_sha256(args.batch_policy)}
    print('Applied throughput policy: '+json.dumps(applied),flush=True)
    return applied


class AudioTargets(Dataset):
    def __init__(self,path,limit=0,max_duration=0):
        self.records=[]
        with Path(path).open() as f:
            for line in f:
                e=json.loads(line)
                if max_duration and e['duration']>max_duration:continue
                self.records.append(e)
                if limit and len(self.records)>=limit:break
        self.lengths=[round(e['duration']*16000) for e in self.records]

    def __len__(self):return len(self.records)

    def __getitem__(self,i):
        e=self.records[i]
        audio,sr=sf.read(e['wav'],dtype='float32')
        if sr!=16000 or audio.ndim!=1:raise ValueError(f'Expected mono 16 kHz: {e["wav"]}')
        audio=(audio-audio.mean())/np.sqrt(audio.var()+1e-7)
        return {'input_values':audio,'targets':e['targets'],'record':e}


def collate(batch):
    audio=[torch.from_numpy(e['input_values']) for e in batch]
    lengths=torch.tensor([len(a) for a in audio])
    values=torch.nn.utils.rnn.pad_sequence(audio,batch_first=True)
    mask=torch.arange(values.shape[1])[None]<lengths[:,None]
    labels={h:torch.nn.utils.rnn.pad_sequence([torch.tensor(e['targets'][h],dtype=torch.long) for e in batch],batch_first=True,padding_value=-100) for h in HEADS}
    return {'input_values':values,'attention_mask':mask.long(),'labels':labels}


def greedy(sequence):
    result=[];prev=None
    for token in sequence:
        if token!=prev and token!=0:result.append(token)
        prev=token
    return result


def boundary_matches(ref,pred,tolerance=.02):
    # Maximal one-to-one tolerance matching of ordered boundary sequences.
    i=j=matched=0
    while i<len(ref) and j<len(pred):
        if abs(ref[i]-pred[j])<=tolerance+1e-9:matched+=1;i+=1;j+=1
        elif ref[i]<pred[j]:i+=1
        else:j+=1
    return matched


class FreezeOTAlignment(TrainerCallback):
    def __init__(self,epoch):self.epoch=epoch
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        if (state.epoch or 0)>=self.epoch:model.freeze_alignment()
    def on_epoch_begin(self,args,state,control,model=None,**kwargs):
        if (state.epoch or 0)>=self.epoch:model.freeze_alignment()


class RuntimeCheckpoint(TrainerCallback):
    """Checkpoint before a short PBS allocation expires, keeping 40-epoch schedule."""
    def __init__(self,seconds):self.seconds=seconds
    def on_train_begin(self,args,state,control,**kwargs):self.started=time.monotonic()
    def on_step_end(self,args,state,control,**kwargs):
        if time.monotonic()-self.started>=self.seconds:
            control.should_save=True
            control.should_training_stop=True


class ThroughputTelemetry(TrainerCallback):
    """Record step wall time and periodic real GPU samples at future launches."""
    def __init__(self,output):
        self.output=output;self.seconds=0.;self.steps=0;self.last_gpu=0.
    def on_step_begin(self,args,state,control,**kwargs):self.started=time.perf_counter()
    def on_step_end(self,args,state,control,**kwargs):
        self.seconds+=time.perf_counter()-self.started;self.steps+=1
    def on_log(self,args,state,control,logs=None,**kwargs):
        if not logs or 'loss' not in logs or not self.steps:return
        record={'optimizer_step':state.global_step,'epoch':state.epoch,
                'mean_train_step_wall_seconds':self.seconds/self.steps,'measured_steps':self.steps,
                'microbatch':args.per_device_train_batch_size,
                'effective_batch':args.per_device_train_batch_size*args.gradient_accumulation_steps}
        record['effective_samples_per_wall_second']=record['effective_batch']/record['mean_train_step_wall_seconds']
        self.seconds=0.;self.steps=0
        now=time.monotonic()
        if torch.cuda.is_available() and now-self.last_gpu>=60:
            try:
                sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                values=sample.stdout.splitlines()[0].split(',') if sample.returncode==0 and sample.stdout.strip() else []
            except (OSError,subprocess.TimeoutExpired):values=[]
            if values:
                for key,value in zip(['gpu_util_percent','gpu_memory_util_percent','gpu_used_MiB','gpu_power_W'],values):
                    try:record[key]=float(value.strip())
                    except ValueError:pass
            record['cuda_peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
            record['cuda_peak_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
            self.last_gpu=now
        with (self.output/'throughput.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if args.report_to and 'wandb' in args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.run.log({'train/global_step':state.global_step,
                               **{'throughput/'+k:v for k,v in record.items() if k!='optimizer_step'}})


class ArchiveCheckpoints(TrainerCallback):
    """Retain model-only weights for later dev selection and averaging."""
    def on_save(self,args,state,control,**kwargs):
        from checkpoint_archive import archive_training
        archive_training(Path(args.output_dir))


class EffectiveBatchSampler:
    """Split every effective batch evenly into accumulation microbatches.

    Keep the final 12 TIMIT examples without leaving an unfinished gradient
    accumulation group. Trainer 4.39 otherwise stops before epoch 40 ends.
    """
    def __init__(self,sampler,size,effective,accumulation):
        self.sampler=sampler;self.size=size;self.effective=effective;self.accumulation=accumulation
        remainder=size%effective
        if remainder and remainder<accumulation:
            raise ValueError('Final effective batch must have at least one sample per microbatch')
    def __len__(self):return math.ceil(self.size/self.effective)*self.accumulation
    def __iter__(self):
        indices=list(self.sampler)
        for start in range(0,len(indices),self.effective):
            group=indices[start:start+self.effective]
            for k in range(self.accumulation):
                yield group[k*len(group)//self.accumulation:(k+1)*len(group)//self.accumulation]


class FourHeadTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.head_sums={h:0. for h in HEADS};self.head_count=0

    def _get_train_sampler(self):
        if self.args.group_by_length:
            return LengthGroupedSampler(self.args.train_batch_size*self.args.gradient_accumulation_steps,
                                        lengths=self.train_dataset.lengths)
        return super()._get_train_sampler()

    def get_train_dataloader(self):
        accumulation=self.args.gradient_accumulation_steps
        batch_sampler=EffectiveBatchSampler(self._get_train_sampler(),len(self.train_dataset),
                                            self.args.train_batch_size*accumulation,accumulation)
        loader=DataLoader(self.train_dataset,batch_sampler=batch_sampler,collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers,pin_memory=self.args.dataloader_pin_memory,
                          worker_init_fn=seed_worker)
        return self.accelerator.prepare(loader)

    def _load_rng_state(self,checkpoint):
        if checkpoint is None:return
        path=Path(checkpoint)/'rng_state.pth'
        if not path.exists():return
        # Trainer 4.39 predates PyTorch 2.6's weights_only default. This file
        # is the local RNG checkpoint produced by this trainer, including NumPy.
        state=torch.load(path,map_location='cpu',weights_only=False)
        random.setstate(state['python']);np.random.set_state(state['numpy'])
        torch.random.set_rng_state(state['cpu'])
        if torch.cuda.is_available() and 'cuda' in state:torch.cuda.random.set_rng_state(state['cuda'])

    def compute_loss(self,model,inputs,return_outputs=False):
        output=model(**inputs)
        if model.training:
            self.head_count+=1
            for h in HEADS:self.head_sums[h]+=output['head_losses'][h].detach().float().item()
        return (output['loss'],output) if return_outputs else output['loss']

    def log(self,logs):
        if 'loss' in logs and self.head_count:
            logs.update({f'head_loss/{h}':self.head_sums[h]/self.head_count for h in HEADS})
            self.head_sums={h:0. for h in HEADS};self.head_count=0
        if 'eval_loss' in logs:
            logs['dev_loss']=logs['eval_loss']
            for name,value in list(logs.items()):
                if name.startswith(('eval_PER_percent/','eval_token_error_percent/')):
                    logs['dev_'+name[5:]]=value
                elif name.startswith('eval_boundary_'):
                    logs['dev_'+name[5:]]=value
            if 'wandb' in self.args.report_to:
                import wandb
                if wandb.run is not None:
                    wandb.run.log({'train/global_step':self.state.global_step,
                                   **{'dev/'+name[4:]:value for name,value in logs.items() if name.startswith('dev_')}})
        super().log(logs)

    def evaluate(self,eval_dataset=None,ignore_keys=None,metric_key_prefix='eval',prediction_path=None):
        data=eval_dataset or self.eval_dataset
        model=self.model
        was_training=model.training;model.eval()
        iterator=DataLoader(data,batch_size=1,num_workers=2,collate_fn=collate)
        loss_sums={h:0. for h in HEADS};refs={h:[] for h in HEADS};hyps={h:[] for h in HEADS}
        boundaries={h:dict(matches=0,reference=0,predicted=0) for h in ['ottc_arpabet','ifmdd_arpabet']}
        start=time.monotonic()
        predictions=[]
        for i,batch in enumerate(iterator):
            batch=self._prepare_inputs(batch)
            with torch.no_grad(),self.compute_loss_context_manager():output=model(**batch)
            length=int(output['lengths'][0])
            e=data.records[i]
            prediction={'utt':e['utt'],'wav':e['wav'],'duration':e['duration'],'heads':{}}
            for h in HEADS:
                path=output['logits'][h][0,:length].argmax(-1).tolist()
                if prediction_path:
                    inverse={v:k for k,v in model.config.vocabs[h].items()}
                    prediction['heads'][h]={'reference':[inverse[t] for t in e['targets'][h]],
                                            'predicted':[inverse[t] for t in greedy(path)],
                                            'frame_token_ids':path,'frame_stride_seconds':.02,
                                            'supervised':bool(e['targets'][h])}
                if not e['targets'][h]:continue
                loss_sums[h]+=float(output['head_losses'][h])
                refs[h].append(' '.join(map(str,e['targets'][h])))
                hyps[h].append(' '.join(map(str,greedy(path))))
                if h in boundaries:
                    reference=e['references'].get(h,[])
                    if reference:
                        end=float(e['duration'])
                        ref=sorted({v for row in reference for v in row[1:] if .02<v<end-.02})
                        pred=[t*.02 for t in range(1,len(path)) if path[t]!=path[t-1] and .02<t*.02<end-.02]
                        counts=boundaries[h]
                        counts['matches']+=boundary_matches(ref,pred)
                        counts['reference']+=len(ref);counts['predicted']+=len(pred)
            if prediction_path:predictions.append(prediction)
        if prediction_path:
            with Path(prediction_path).open('w') as f:
                for prediction in predictions:f.write(json.dumps(prediction,ensure_ascii=False)+'\n')
        metrics={f'{metric_key_prefix}_loss':sum(loss_sums[h]/len(refs[h]) if refs[h] else 0. for h in HEADS)/len(HEADS),
                 f'{metric_key_prefix}_runtime':time.monotonic()-start}
        for h in HEADS:
            errors=jiwer.process_words(refs[h],hyps[h]).wer if refs[h] else 0.
            metrics[f'{metric_key_prefix}_head_loss/{h}']=loss_sums[h]/len(refs[h]) if refs[h] else 0.
            metrics[f'{metric_key_prefix}_supervised_utterances/{h}']=len(refs[h])
            metric='token_error_percent' if h=='zipa_ipa' else 'PER_percent'
            metrics[f'{metric_key_prefix}_{metric}/{h}']=100*errors
        for h,c in boundaries.items():
            denom=c['reference']+c['predicted']
            metrics[f'{metric_key_prefix}_boundary_F1_20ms/{h}']=200*c['matches']/denom if denom else 0.
            metrics[f'{metric_key_prefix}_boundary_precision_20ms/{h}']=100*c['matches']/c['predicted'] if c['predicted'] else 0.
            metrics[f'{metric_key_prefix}_boundary_recall_20ms/{h}']=100*c['matches']/c['reference'] if c['reference'] else 0.
            metrics[f'{metric_key_prefix}_boundary_R_value_20ms/{h}']=r_value_percent(
                metrics[f'{metric_key_prefix}_boundary_precision_20ms/{h}'],
                metrics[f'{metric_key_prefix}_boundary_recall_20ms/{h}'])
        # IPA127 is token error rate. No fabricated reference time metrics.
        metrics[f'{metric_key_prefix}_IPA127_token_error_percent']=metrics[f'{metric_key_prefix}_token_error_percent/zipa_ipa']
        self.log(metrics)
        self.control=self.callback_handler.on_evaluate(self.args,self.state,self.control,metrics)
        if was_training:model.train()
        return metrics


def main():
    from reproduction.wandb_policy import log_evaluation_artifact
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--objective',choices=['ctc','ce','jsd_upstream','jsd_standard'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--freeze-alignment-epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--effective-batch-size',type=int,default=64)
    parser.add_argument('--batch-policy',type=Path,default=ROOT/'reproduction/english_suite/libri_batch_policy.json')
    parser.add_argument('--max-steps',type=int,default=-1)
    parser.add_argument('--max-runtime-seconds',type=int,default=0)
    parser.add_argument('--train-limit',type=int,default=0)
    parser.add_argument('--eval-limit',type=int,default=0)
    parser.add_argument('--max-duration',type=float,default=20)
    parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--wandb-mode',choices=['online','offline','disabled'],default=os.environ.get('WANDB_MODE','online'))
    args=parser.parse_args();batch_policy=apply_batch_policy(args);set_seed(42);torch.set_num_threads(4)
    from accelerate import PartialState
    runtime=PartialState()
    if runtime.num_processes!=1:raise RuntimeError('Each objective must run independently with world_size=1')
    print(f'Independent training: world_size={runtime.num_processes}, device={runtime.device}',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    vocabs=json.loads((args.data/'vocabs.json').read_text())
    metadata=json.loads((args.data/'metadata.json').read_text())
    train=AudioTargets(args.data/'train-train.jsonl',args.train_limit,args.max_duration)
    dev=AudioTargets(args.data/'train-dev.jsonl',args.eval_limit,args.max_duration)
    accumulation=args.effective_batch_size//args.batch_size
    if accumulation*args.batch_size!=args.effective_batch_size:raise ValueError('Effective batch must be divisible by microbatch')
    if args.tiny:
        base=Wav2Vec2Config(hidden_size=32,num_hidden_layers=1,num_attention_heads=2,intermediate_size=64,
                           conv_dim=(8,8,8),conv_stride=(5,4,4),conv_kernel=(10,8,4),
                           num_conv_pos_embeddings=16,num_conv_pos_embedding_groups=2,mask_time_prob=0.)
    else:base=Wav2Vec2Config.from_pretrained(ROOT/'reproduction/models/wav2vec2-large-lv60',local_files_only=True)
    config=MultiPhoneConfig(encoder_config=base.to_dict(),vocabs=vocabs,objective=args.objective)
    checkpoints=[p for p in args.output.glob('checkpoint-*') if p.name[11:].isdigit()]
    latest=max(checkpoints,key=lambda p:int(p.name[11:]),default=None)
    if latest:
        model=MultiPhoneModel.from_pretrained(latest)
        if model.config.vocabs!=vocabs or model.config.objective!=args.objective:raise ValueError('Checkpoint configuration differs')
    else:
        model=MultiPhoneModel(config)
        if not args.tiny:
            state=torch.load(ROOT/'reproduction/models/wav2vec2-large-lv60/pytorch_model.bin',map_location='cpu',weights_only=False)
            state={k.removeprefix('wav2vec2.'):v for k,v in state.items() if k.startswith('wav2vec2.')}
            info=model.encoder.load_state_dict(state,strict=False)
            if info.missing_keys or info.unexpected_keys:raise ValueError(str(info))
    model.encoder.feature_extractor._freeze_parameters()
    config_dict={**vars(args),'suffix':args.output.name,'dataset':metadata['dataset'],'loss_type':args.objective,
                 'continue_if_exist':True,'wandb_project':'ottc-english-multiphoneset','wandb_entity':'kevingenghaopeng-',
                 'learning_rate':5e-5 if args.objective=='ctc' else 2e-4,'gradient_accumulation_steps':accumulation,
                 'head_loss_reduction':'utterance mean, equally weighted heads','phone_metadata':metadata,
                 'hostname':platform.node(),'job':os.environ.get('PBS_JOBID'),'seed':42,
                 'microbatch_optimization':batch_policy,
                 'segmentation_R_value':'Frozen IF-MDD formula applied to current P/R; existing boundary protocol unchanged',
                 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'reproduction/multihead.py',ROOT/'reproduction/prepare_multihead.py',ROOT/'reproduction/prepare_multihead_libri.py',ROOT/'reproduction/segmentation_metrics.py',ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py']}}
    config_dict['base_encoder_sha256']=file_sha256(ROOT/'reproduction/models/wav2vec2-large-lv60/pytorch_model.bin') if not args.tiny else 'tiny-validation'
    config_dict['data_sha256']={p.name:file_sha256(p) for p in args.data.glob('*.json*')}
    source_snapshot=args.output/'source_snapshot';source_snapshot.mkdir(exist_ok=True)
    import shutil
    # Preserve the launch configuration and code of every earlier segment.
    if (args.output/'run_config.json').exists():
        history=args.output/'launch_history'/str(time.time_ns());history.mkdir(parents=True)
        shutil.copy2(args.output/'run_config.json',history/'run_config.json')
        shutil.copytree(source_snapshot,history/'source_snapshot')
    for name in ['train_multihead.py','multihead.py','prepare_multihead.py','prepare_multihead_libri.py','segmentation_metrics.py']:
        shutil.copy2(ROOT/'reproduction'/name,source_snapshot/name)
    shutil.copy2(ROOT/'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py',source_snapshot/'boundary_rvalue.py')
    config_dict={k:str(v) if isinstance(v,Path) else v for k,v in config_dict.items()}
    (args.output/'run_config.json').write_text(json.dumps(config_dict,ensure_ascii=False,indent=2)+'\n')
    (args.output/'dataset_sizes.json').write_text(json.dumps({'train':len(train),'dev':len(dev)}))
    if args.wandb_mode!='disabled':
        run=start_training_run(config_dict,args.output)
        run.config.update(config_dict,allow_val_change=True)
    if args.tiny:
        # The tiny smoke verifies execution, not speech model quality.
        model.encoder.config.mask_time_prob=0.
    training_args=TrainingArguments(output_dir=str(args.output),num_train_epochs=args.epochs,
                                   max_steps=args.max_steps,per_device_train_batch_size=args.batch_size,
                                   gradient_accumulation_steps=accumulation,per_device_eval_batch_size=1,
                                   learning_rate=config_dict['learning_rate'],weight_decay=.005,warmup_steps=500,
                                   bf16=torch.cuda.is_available(),fp16=False,gradient_checkpointing=True,
                                   logging_steps=50 if args.max_steps<0 else 1,evaluation_strategy='epoch',save_strategy='epoch',
                                   save_total_limit=3,save_safetensors=False,group_by_length=True,
                                   dataloader_num_workers=4,remove_unused_columns=False,
                                   report_to=['wandb','tensorboard'] if args.wandb_mode!='disabled' else [],
                                   run_name=args.output.name,seed=42,ddp_find_unused_parameters=False)
    callbacks=[ThroughputTelemetry(args.output),ArchiveCheckpoints()]
    if args.objective!='ctc':callbacks.append(FreezeOTAlignment(args.freeze_alignment_epochs))
    if args.max_runtime_seconds:callbacks.append(RuntimeCheckpoint(args.max_runtime_seconds))
    if args.wandb_mode!='disabled':callbacks.append(LossCurveCallback())
    trainer=FourHeadTrainer(model=model,args=training_args,train_dataset=train,eval_dataset=dev,data_collator=collate,
                            callbacks=callbacks)
    try:
        if not args.evaluate_only:
            trainer.train(resume_from_checkpoint=str(latest) if latest else None)
            if args.max_steps<0 and (trainer.state.epoch or 0)<args.epochs-1e-6:
                trainer.save_state()
                (args.output/'CHECKPOINT_PAUSED.json').write_text(json.dumps(
                    {'objective':args.objective,'steps':trainer.state.global_step,'epochs':trainer.state.epoch,
                     'reason':'Allocation runtime budget; resume to complete requested epochs'},indent=2)+'\n')
                if args.wandb_mode!='disabled':
                    import wandb
                    wandb.run.summary.update({'training/complete':False,'training/checkpointed_for_continuation':True})
                finish_training_run()
                return
            trainer.save_model(str(args.output/'final'))
            trainer.save_state()
        metrics=trainer.evaluate(prediction_path=args.output/'dev_predictions.jsonl')
        (args.output/'dev_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if args.max_steps<0:
            test=AudioTargets(args.data/'test.jsonl',args.eval_limit,args.max_duration)
            metrics=trainer.evaluate(test,metric_key_prefix='test',prediction_path=args.output/'test_predictions.jsonl')
            (args.output/'test_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        if args.wandb_mode!='disabled':
            import wandb
            wandb.run.summary.update({'benchmark/'+k:v for k,v in metrics.items()})
            artifact=wandb.Artifact(args.output.name+'-final-benchmark',type='evaluation')
            for name in ['dev_metrics.json','test_metrics.json','dev_predictions.jsonl','test_predictions.jsonl','run_config.json']:
                path=args.output/name
                if path.exists():artifact.add_file(str(path),name=name)
            log_evaluation_artifact(wandb.run, artifact)
        (args.output/'TRAIN_COMPLETE.json').write_text(json.dumps({'objective':args.objective,'steps':trainer.state.global_step,
                                                                'epochs':trainer.state.epoch,'smoke':args.max_steps>0},indent=2))
        (args.output/'CHECKPOINT_PAUSED.json').unlink(missing_ok=True)
        if args.wandb_mode!='disabled':
            wandb.run.summary.update({'training/complete':True,'training/checkpointed_for_continuation':False})
    except Exception:
        finish_training_run(1)
        raise
    finish_training_run()


if __name__=='__main__':main()
