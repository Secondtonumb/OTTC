"""Five-epoch, three-cost Libri960 continuation of the matched Libri100 models."""
import bootstrap
from bootstrap import ROOT,HERE
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import datasets
import torch
from transformers import Trainer,TrainerCallback,TrainingArguments,Wav2Vec2Processor,set_seed
from transformers.trainer_callback import PrinterCallback
from ottc.models.sequence.wav2vec2_ottc import wvForOTTC
from ottc.tools.data_collator.data_collator import DataCollatorCTCWithPadding
from ottc.utils.wandb_tracking import defaults,start_training_run,finish_training_run,LossCurveCallback
from shared import boundary_matches,phone_per,preprocess,recognition_metrics,DevelopmentSegmentation
from data import load_data,check_normalization
from archive import EpochArchive,BestDev,atomic_json,digest

OBJECTIVES={'ce':'crossentropy','jsd_upstream':'jsd','jsd_standard':'jsd_standard'}
PARENT=ROOT/'reproduction/models/large_models_results/wav2vec2-large-lv60'
OFFSET=17840


def model_directory(objective,smoke=False):
    return HERE/'models'/(('smoke_' if smoke else '')+objective)


def latest_checkpoint(output):
    candidates=[p for p in Path(output).glob('checkpoint-*') if p.name.split('-')[-1].isdigit() and (p/'trainer_state.json').exists()]
    return max(candidates,key=lambda p:int(p.name.split('-')[-1]),default=None)


def hashes():
    files=list(HERE.glob('*.py'))+[ROOT/'ottc/models/sequence/wav2vec2_ottc.py',ROOT/'ottc/ops_loss/ot_loss.py',
        ROOT/'ottc/utils/wandb_tracking.py',ROOT/'reproduction/libri100_singlehead_20261014/train.py',
        ROOT/'reproduction/segmentation_metrics.py']
    return {str(p):digest(p) for p in sorted(files)}


def parent_checkpoint(objective):
    directory=PARENT/f'runexp_libri100_singlehead_{objective}_20261014'
    complete=json.loads((directory/'TRAIN_COMPLETE.json').read_text())
    assert complete['epoch']==40 and complete['global_step']==OFFSET
    checkpoint=Path(complete['checkpoint'])
    config=json.loads((checkpoint/'config.json').read_text())
    assert config['bins_metric']==OBJECTIVES[objective] and config['vocab_size']==42
    assert config['forward_type']=='regfrozen' and 0<config['frozen_steps_for_regfrozen']<OFFSET
    prior=json.loads((directory/'experiment.json').read_text())
    for relative in ['ottc/models/sequence/wav2vec2_ottc.py','ottc/ops_loss/ot_loss.py']:
        path=ROOT/relative
        assert digest(path)==prior['source_sha256'][str(path)],'Original cost/model source changed'
    return directory,checkpoint


class Progress(TrainerCallback):
    def __init__(self,seconds,smoke,stop_at_two,steps_per_epoch):
        self.seconds=seconds;self.smoke=smoke;self.stop_at_two=stop_at_two
        self.steps_per_epoch=steps_per_epoch
        self.interior_saves={math.ceil(steps_per_epoch/3),math.ceil(2*steps_per_epoch/3)}

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.begin=time.monotonic();self.start=state.global_step
        model.step=OFFSET+state.global_step-1

    def on_step_begin(self,args,state,control,model=None,**kwargs):
        model.step=OFFSET+state.global_step-1

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step%self.steps_per_epoch in self.interior_saves:control.should_save=True
        if self.stop_at_two and state.global_step>=2:
            control.should_training_stop=True;control.should_save=True
        elapsed=time.monotonic()-self.begin;advanced=state.global_step-self.start
        if advanced in [1,5,50,200] or advanced%1000==0:
            record=dict(stage_step=state.global_step,cumulative_optimizer_step=OFFSET+state.global_step,
                stage_epoch=state.epoch,elapsed_seconds=elapsed,seconds_per_step=elapsed/advanced)
            print('TRAIN_PROGRESS',json.dumps(record),flush=True)
        if elapsed>=self.seconds:
            control.should_training_stop=True;control.should_save=True
        return control

    def on_log(self,args,state,control,logs=None,**kwargs):
        import wandb
        if wandb.run:
            wandb.run.log({'train/global_step':state.global_step,'continuation/cumulative_optimizer_step':OFFSET+state.global_step,
                'continuation/cumulative_epoch':40+(state.epoch or 0)})


class ContinuationTrainer(Trainer):
    def __init__(self,*a,parent_optimizer=None,resuming=False,validation=False,**kw):
        self.parent_optimizer=parent_optimizer;self.resuming=resuming;self.validation=validation
        self.validation_batches=[];self.parent_optimizer_restored=False
        super().__init__(*a,**kw)

    def create_optimizer(self):
        result=super().create_optimizer()
        if not self.resuming and not self.parent_optimizer_restored:
            saved=torch.load(self.parent_optimizer,map_location='cpu',weights_only=False)
            assert [len(g['params']) for g in saved['param_groups']]==[len(g['params']) for g in self.optimizer.param_groups]
            sample_key=next(k for k,v in saved['state'].items() if 'exp_avg' in v)
            moment=saved['state'][sample_key]['exp_avg'].flatten()[:16].clone()
            self.optimizer.load_state_dict(saved)
            restored=self.optimizer.state_dict()['state'][sample_key]['exp_avg'].flatten()[:16].detach().cpu()
            torch.testing.assert_close(moment,restored,rtol=0,atol=0)
            for group in self.optimizer.param_groups:
                group['lr']=self.args.learning_rate;group['initial_lr']=self.args.learning_rate
            self.parent_optimizer_restored=True
            print('PARENT_ADAMW_MOMENTS_RESTORED; NEW_FIVE_EPOCH_LR_SCHEDULE',flush=True)
            del saved
        return result

    def training_step(self,model,inputs):
        if not self.validation:return super().training_step(model,inputs)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        loss=super().training_step(model,inputs)
        torch.cuda.synchronize();raw=self.accelerator.unwrap_model(model)
        assert torch.isfinite(loss)
        for p in [raw.lm_head.weight,raw.wav2vec2.encoder.layers[0].attention.q_proj.weight]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
        assert all(p.grad is None for p in raw.wav2vec2.feature_extractor.parameters())
        assert all(p.grad is None for p in raw.wav2vec2_freeze.parameters())
        assert raw.lm_weight.weight.grad is None,'Continue the already frozen alignment head'
        record=dict(model_step=raw.step,loss=float(loss),batch_size=len(inputs['input_values']),
            padded_seconds=inputs['input_values'].shape[1]/16000,seconds=time.monotonic()-begin,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            pretrained_transformer_gradient_finite=True,cnn_frozen=True,alignment_remains_frozen=True)
        self.validation_batches.append(record);print('GPU_VALIDATION',json.dumps(record),flush=True)
        return loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=list(OBJECTIVES),required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-validation',action='store_true')
    parser.add_argument('--max-runtime-seconds',type=float,default=81000)
    args=parser.parse_args();assert not args.resume_validation or args.smoke
    assert torch.cuda.is_available(),'Use a Miyabi GPU allocation'
    torch.set_num_threads(4);set_seed(42)
    output=model_directory(args.objective,args.smoke);output.mkdir(parents=True,exist_ok=True)
    if (output/'TRAIN_COMPLETE.json').exists():print('Completed continuation reused',flush=True);return
    if (output/'VALIDATION_COMPLETE.json').exists() and args.smoke:
        assert json.loads((output/'VALIDATION_COMPLETE.json').read_text())['source_sha256']==hashes()
        print('Completed four-step optimizer/epoch validation reused',flush=True);return
    source_hashes=hashes()
    if not args.smoke:
        gate=json.loads((HERE/'validation'/args.objective/'PASS.json').read_text())
        assert gate['source_sha256']==source_hashes,'Sources changed since validation'
    original,parent=parent_checkpoint(args.objective)
    audit=json.loads((HERE/'data/PREPARE_COMPLETE.json').read_text())
    parent_sha=digest(parent/'model.safetensors')
    identity=dict(objective=args.objective,parent_checkpoint=str(parent),parent_weights_sha256=parent_sha,
        inventory_sha256=audit['phone_inventory_sha256'],source_sha256=source_hashes,stage_epochs=5,
        dataset_sha256=audit['compact_sha256'],learning_rate=2e-5,smoke=args.smoke)
    if (output/'experiment.json').exists():
        assert json.loads((output/'experiment.json').read_text())['identity']==identity
    else:
        atomic_json(output/'experiment.json',dict(identity=identity,dataset='full Libri960 continuation from Libri100 epoch40',
            utterances=audit['utterances'],hours=audit['hours'],effective_batch_size=64,precision='BF16 autocast; FP32 weights',
            optimizer='inherit parent AdamW moments; new 5-epoch linear schedule, lr2e-5, warmup500',
            freeze='retain epoch40 frozen alignment backbone/head; CNN frozen; phone Transformer fine-tuned',
            selection='best development PER including parent baseline; final5 and last-five average also evaluated',
            epoch_retention='save full checkpoints at 1/3, 2/3 and epoch end; retain all15 over5epochs; atomically retain all5 completed-epoch weights separately',
            parent_wandb=json.loads((original/'wandb_run.json').read_text()),host=platform.node(),job=os.environ.get('PBS_JOBID')))
        snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for path in HERE.glob('*.py'):(snapshot/path.name).write_bytes(path.read_bytes())
    data=load_data(args.smoke)
    processor=Wav2Vec2Processor.from_pretrained(HERE/'data/processor')
    processor.save_pretrained(output/'processor')
    (output/'vocab_used.json').write_bytes((HERE/'data/vocab.json').read_bytes())
    atomic_json(output/'dataset_sizes.json',{k:len(v) for k,v in data.items()})
    local=latest_checkpoint(output)
    if args.resume_validation:assert local is not None and int(local.name.split('-')[-1])==2
    model,loading=wvForOTTC.from_pretrained(local or parent,output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    assert model.config.bins_metric==OBJECTIVES[args.objective]
    model.freeze_feature_encoder();model.step=OFFSET+(int(local.name.split('-')[-1]) if local else 0)-1
    assert model.step>model.frozen_steps_for_regfrozen
    options=defaults()
    config=dict(dataset='libri960_continue5_20261017',suffix='libri960-continue5-'+args.objective+('-smoke' if args.smoke else ''),
        loss_type='ottc',bins_metric=OBJECTIVES[args.objective],continue_if_exist=True,
        wandb_mode=os.environ.get('WANDB_MODE',options['mode']),wandb_project=os.environ.get('WANDB_PROJECT',options['project']),
        wandb_entity=os.environ.get('WANDB_ENTITY',options['entity']),identity=identity,epochs=5,batch_size=64,
        learning_rate=2e-5,warmup_steps=500,weight_decay=.005,seed=42,stage_offset_steps=OFFSET,
        corpus=audit,source_sha256=source_hashes)
    atomic_json(output/'run_config.json',config)
    start_training_run(config,output)
    training_args=TrainingArguments(output_dir=str(output),per_device_train_batch_size=64,
        per_device_eval_batch_size=8,num_train_epochs=5,max_steps=4 if args.smoke else -1,
        learning_rate=2e-5,warmup_steps=500,weight_decay=.005,fp16=False,bf16=True,
        evaluation_strategy='epoch',save_strategy='no',save_total_limit=None,
        logging_steps=1 if args.smoke else 50,report_to=['tensorboard','wandb'],run_name=config['suffix'],
        dataloader_num_workers=4,group_by_length=True,length_column_name='length',remove_unused_columns=False,
        disable_tqdm=True,seed=42)
    parent_metric=json.loads((original/'eval_results.json').read_text())['eval_nosil_PER_percent']
    trainer=ContinuationTrainer(model=model,args=training_args,train_dataset=data['train'],eval_dataset=data['dev'],
        data_collator=DataCollatorCTCWithPadding(processor=processor),tokenizer=processor.feature_extractor,
        compute_metrics=recognition_metrics(processor),preprocess_logits_for_metrics=preprocess,
        parent_optimizer=parent/'optimizer.pt',resuming=local is not None,validation=args.smoke,
        callbacks=[LossCurveCallback(),EpochArchive(output,identity),BestDev(output,parent,parent_metric),
            DevelopmentSegmentation(processor,data['dev'],output,args.smoke),
            Progress(args.max_runtime_seconds,args.smoke,args.smoke and not args.resume_validation,
                     math.ceil(len(data['train'])/64))])
    if args.smoke:
        normalization=check_normalization()
        previous=json.loads((output/'validation_batches.json').read_text()) if (output/'validation_batches.json').exists() else []
    try:
        trainer.train(resume_from_checkpoint=str(local) if local else None)
        trainer.save_state()
        checkpoint=latest_checkpoint(output)
        assert checkpoint is not None and int(checkpoint.name.split('-')[-1])==trainer.state.global_step
        if args.smoke:
            records=previous+trainer.validation_batches
            atomic_json(output/'validation_batches.json',records)
            if trainer.state.global_step==4:
                assert len(records)==4 and all((output/'epoch_weights'/f'epoch-{e:03d}'/'EPOCH_COMPLETE.json').exists() for e in range(1,5))
                assert (output/'best-dev/model.safetensors').exists()
                atomic_json(output/'VALIDATION_COMPLETE.json',dict(source_sha256=source_hashes,parent_weights_sha256=parent_sha,
                    optimizer_and_rng_resume_verified=True,normalization=normalization,batches=records,epoch_archives=4,
                    wandb=json.loads((output/'wandb_run.json').read_text())))
            else:assert trainer.state.global_step==2
        else:
            expected=5*math.ceil(audit['utterances']/64)
            if trainer.state.global_step!=expected:
                atomic_json(output/'PAUSED.json',dict(epoch=trainer.state.epoch,stage_step=trainer.state.global_step,checkpoint=str(checkpoint)))
                finish_training_run();raise SystemExit(75)
            assert trainer.state.epoch==5
            for epoch in range(1,6):assert (output/'epoch_weights'/f'epoch-{epoch:03d}'/'EPOCH_COMPLETE.json').exists()
            atomic_json(output/'TRAIN_COMPLETE.json',dict(epoch=5,global_step=expected,cumulative_optimizer_step=OFFSET+expected,
                checkpoint=str(checkpoint),objective=args.objective,parent_checkpoint=str(parent),parent_sha256=parent_sha))
        assert digest(parent/'model.safetensors')==parent_sha,'Parent checkpoint must remain unchanged'
    except BaseException:
        finish_training_run(exit_code=1);raise
    else:finish_training_run()


if __name__=='__main__':main()
