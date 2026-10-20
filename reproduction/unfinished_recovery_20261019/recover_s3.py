"""Reconstruct quota-lost optimizer/RNG state by verifying a first-epoch replay."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ORIGINAL=ROOT/'reproduction/timit_ssl_native10_20261019'
sys.path[:0]=[str(ORIGINAL),str(ROOT/'reproduction/vendor'),str(ROOT/'reproduction'),str(ROOT)]
import torch
from transformers import TrainingArguments,TrainerCallback,set_seed
from common import recipe,load_data,build_model,Collator,source_hashes,write,sha
from reproduction.timit_ssl_native10_20261019.train import QuietTrainer,Policy,ChunkedCollator,compute_metrics


class NoTracking:
    def log(self,*args,**kwargs):pass


class StopAfterEpochOne(TrainerCallback):
    def on_epoch_end(self,args,state,control,**kwargs):
        if state.epoch>=1.-1e-8:control.should_training_stop=True;control.should_save=True
        return control


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True);args=parser.parse_args()
    assert args.recipe in ['wav2vec2__timit__10ms__s3t2ss__ce','wav2vec2__timit__10ms__s3t2ss__jsd_standard']
    assert torch.cuda.is_available();torch.set_num_threads(4)
    output=ORIGINAL/'models'/args.recipe
    lock=(output/'.training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=output/'epoch_weights/epoch-001.pt';assert target.exists()
    destination=output/'checkpoint-73'
    if (output/'RECOVERY_PASS.json').exists():
        saved=json.loads((output/'RECOVERY_PASS.json').read_text())
        assert saved['exact_model_match'] and (destination/'optimizer.pt').exists() and (destination/'rng_state.pth').exists()
        print('RECOVERY_ALREADY_VERIFIED',output,flush=True);return
    assert destination.exists() and not (destination/'optimizer.pt').exists() and not (destination/'trainer_state.json').exists()
    original_sources=source_hashes();cfg=recipe(args.recipe)
    data,rows,vocab,processor,audit=load_data(cfg);cfg['data']=audit
    profile=json.loads((ORIGINAL/'profiles'/args.recipe/'PASS.json').read_text())
    assert profile['source_sha256']==original_sources
    micro=profile['microbatch'];steps=math.ceil(len(data['train'])/64)
    set_seed(42);model,topo=build_model(cfg,vocab,steps)
    cfg.update(microbatch=micro,accumulation_steps=profile['accumulation_steps'],planned_steps_per_epoch=steps)
    assert cfg==json.loads((output/'run_config.json').read_text()),'Original source/data/model contract changed'
    attempt='replay_'+os.environ.get('PBS_JOBID',str(int(time.time())))
    replay=HERE/'s3_replays'/args.recipe/attempt;replay.mkdir(parents=True,exist_ok=False)
    processor.save_pretrained(replay/'processor')
    training_args=TrainingArguments(output_dir=str(replay),per_device_train_batch_size=64,per_device_eval_batch_size=micro,
        gradient_accumulation_steps=1,num_train_epochs=40,learning_rate=cfg['learning_rate'],
        warmup_steps=cfg['warmup_steps'],weight_decay=.005,bf16=True,fp16=False,seed=42,data_seed=42,
        evaluation_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=False,
        logging_steps=10,report_to=[],dataloader_num_workers=4,dataloader_pin_memory=True,
        remove_unused_columns=False,group_by_length=True,length_column_name='length',max_grad_norm=1.,
        max_steps=-1,disable_tqdm=True,save_safetensors=True)
    policy=Policy(replay,cfg,rows['dev'],vocab,processor,topo,float('inf'),NoTracking())
    trainer=QuietTrainer(model=model,args=training_args,train_dataset=data['train'],eval_dataset=data['dev'],
        data_collator=ChunkedCollator(Collator(processor,topo['mapping']),micro),tokenizer=processor,
        compute_metrics=compute_metrics(vocab,topo),callbacks=[policy,StopAfterEpochOne()])
    print('REPLAY_FIRST_EPOCH_FULL40_SCHEDULE',args.recipe,'micro',micro,flush=True)
    trainer.train(ignore_keys_for_eval=['hidden_states','attentions'])
    assert trainer.state.global_step==73 and trainer.state.epoch==1.
    stored=torch.load(target,map_location='cpu',weights_only=False,mmap=True)
    assert stored['epoch']==1 and stored['step']==73 and stored['recipe']==args.recipe
    current=model.state_dict();assert set(current)==set(stored['model'])
    differences={}
    for key,value in current.items():
        reference=stored['model'][key];now=value.detach().cpu()
        if not torch.equal(now,reference):differences[key]=float((now-reference).abs().max())
    maximum=max(differences.values(),default=0.)
    check=dict(recipe=args.recipe,reference=str(target),reference_sha256=sha(target),replay=str(replay),
        epoch=1,optimizer_step=73,exact_model_match=not differences,model_max_absolute_difference=maximum,
        mismatched_tensor_count=len(differences),largest_differences=sorted(differences.items(),key=lambda x:x[1],reverse=True)[:12],
        source_sha256=original_sources,source_unchanged=source_hashes()==original_sources,
        production_wandb_run_id=json.loads((output/'wandb_run.json').read_text())['id'])
    write(replay/'REPLAY_COMPARISON.json',check);print('REPLAY_COMPARISON',json.dumps(check),flush=True)
    assert check['source_unchanged'] and check['exact_model_match'],'Replay differs: original artifacts preserved; not claiming an exact recovery'
    checkpoint=replay/'checkpoint-73'
    for name in ['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth']:
        assert (checkpoint/name).is_file() and (checkpoint/name).stat().st_size>0,name
    stage=output/'restored_checkpoint-73.tmp';assert not stage.exists()
    shutil.copytree(checkpoint,stage)
    preserve=output/'recovery_preserved'/attempt;preserve.mkdir(parents=True,exist_ok=False)
    destination.rename(preserve/'partial_checkpoint-73')
    stage.rename(destination)
    check.update(recovered_checkpoint=str(destination),original_partial_checkpoint_preserved=str(preserve/'partial_checkpoint-73'),
        optimizer_scheduler_rng='Reconstructed with bitwise-identical first-epoch model under original40epoch schedule',
        original_epoch_weight_preserved_sha256=sha(target),recovery_did_not_log_replayed_epoch_to_production_WB=True)
    assert check['reference_sha256']==check['original_epoch_weight_preserved_sha256']
    write(output/'RECOVERY_PASS.json',check)
    print('FULL_RESUME_RECOVERED',destination,flush=True)


if __name__=='__main__':main()
