"""Independent CE stabilization stage, frequent recoverable checkpoints and TIMIT curves."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd')
from framece_model import FrameCE, targets
from prepare import sha
from framece_metrics import evaluate
import microbatching
import step_checkpoint

ORIGIN = Path('/work/gm64/m64000/fa-bench/data/mfa_spottc_jsd_s3t2_10ms_20261009/training/step-000068868.pt')


class Schedule:
    def __init__(self, rows, rank, world, batch_size):
        self.rows, self.rank, self.world, self.batch_size = rows, rank, world, batch_size
        self.epoch, self.cursor = 0, 0

    def batches(self):
        order = torch.randperm(len(self.rows),generator=torch.Generator().manual_seed(731+self.epoch)).tolist()
        count = math.ceil(len(order)/self.world)*self.world
        order += order[:count-len(order)]
        local = order[self.rank::self.world]
        return [local[i:i+self.batch_size] for i in range(0,len(local),self.batch_size)]

    def state_dict(self):
        return dict(epoch=self.epoch,cursor=self.cursor,rank=self.rank,world=self.world,batch_size=self.batch_size,seed=731)

    def load_state_dict(self, state):
        assert state['rank']==self.rank and state['world']==self.world and state['batch_size']==self.batch_size
        self.epoch, self.cursor = state['epoch'], state['cursor']


class Features(Dataset):
    def __init__(self, rows): self.rows, self.caches = rows, {}
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        r = self.rows[index]; path = r['feature_file']
        if path not in self.caches: self.caches[path] = np.memmap(path,mode='r',dtype=np.float32)
        x = self.caches[path][r['feature_offset']:r['feature_offset']+80*r['frames']]
        return r, torch.from_numpy(np.array(x).reshape(-1,80))


def collate(items):
    rows, xs = zip(*items)
    return list(rows), torch.nn.utils.rnn.pad_sequence(xs,batch_first=True), torch.tensor([len(x) for x in xs])


def open_wandb(output, config, smoke):
    import wandb
    defaults = json.loads((HERE.parent/'wandb_config.json').read_text())
    identity = output/'wandb_run.json'
    saved = json.loads(identity.read_text()) if identity.exists() else {}
    opts = dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),
                entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
                mode='offline' if smoke else os.environ.get('WANDB_MODE',defaults.get('mode','online')),
                name=output.name,group='spottc_jsd_s3t2_libri100_framece_20261012',
                id=saved.get('id'),resume='allow' if saved else None,dir=str(output),config=config,
                settings=wandb.Settings(init_timeout=45,disable_code=True))
    try: run = wandb.init(**opts)
    except wandb.errors.CommError:
        opts.update(mode='offline'); opts.pop('resume',None); run = wandb.init(**opts)
    identity.write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode))+'\n')
    run.define_metric('optimizer_step')
    for prefix in ('train/*','timit_dev/*','timit_test/*'):
        run.define_metric(prefix,step_metric='optimizer_step')
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--save-every',type=int,default=50)
    ap.add_argument('--eval-every',type=int,default=50); ap.add_argument('--max-wall-seconds',type=int,default=13800)
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--stop-after',type=int,default=0)
    a = ap.parse_args(); started = time.monotonic()
    rank, world, local = [int(os.environ.get(k,d)) for k,d in [('RANK','0'),('WORLD_SIZE','1'),('LOCAL_RANK','0')]]
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.set_num_threads(4); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    a.output.mkdir(parents=True,exist_ok=True)
    if rank == 0:
        from validate import main as validate_historical_ce
        (HERE/'validation').mkdir(exist_ok=True)
        validate_historical_ce()
    dist.barrier()
    manifest = HERE/'data/training.jsonl'
    ready = json.loads((HERE/'data/PREPARE_COMPLETE.json').read_text())
    assert sha(manifest)==ready['training_manifest_sha256']
    rows = [json.loads(line) for line in manifest.open()]
    assert len(rows)==28538
    if a.smoke: rows=rows[:max(64,world*a.batch_size*3)]
    net = FrameCE()
    initial = torch.load(ORIGIN,map_location='cpu',mmap=True,weights_only=False)
    assert initial['contract']['topology']=='s3_t2' and initial['contract']['ottc_cost']=='jsd_upstream'
    net.load_state_dict(initial['model'],strict=True)
    origin_step = initial['progress']['global_step']; del initial
    net.cuda(); model=DDP(net,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=a.lr)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    schedule=Schedule(rows,rank,world,a.batch_size)
    source_paths=[HERE/n for n in ['train_framece.py','framece_model.py','framece_metrics.py','prepare.py','historical_codec.py','historical_times.py']]
    source_paths += [Path('/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd/spottc_jsd_model.py'),
                     HERE.parent/'english_suite/mfa_metrics.py',
                     Path('/work/gm64/m64000/fa-bench/analysis/mfa_spottc_jsd_s3t2ss/native_alignment.py'),
                     Path(step_checkpoint.__file__), Path(microbatching.__file__)]
    contract=dict(stage='libri100_framewise_phone_ce_only',origin_checkpoint=str(ORIGIN),
        origin_progress_step=origin_step,origin_checkpoint_sha256=sha(ORIGIN),
        training_manifest_sha256=sha(manifest),timeline_sha256=sha(HERE/'timeline_provenance.json'),
        source_hashes={str(p):sha(p) for p in source_paths},
        epochs=a.epochs,batch_size_per_rank=a.batch_size,world_size=world,lr=a.lr,optimizer='Adam',
        ce_weight=1.,sequence_jsd_weight=0.,topology_forward_sum_weight=0.,topology='s3_t2',
        state_posterior='logsumexp of existing three states per phone; blank kept in normalization',
        weight_head_frozen=True,backbone_trainable=True,save_every=a.save_every,eval_every=a.eval_every,
        timit_dev_utterances=400,timit_test_utterances=192,timit_gradient_training=False,
        historical_backfill=False,smoke=a.smoke,precision='bf16',sampler_seed=731,
        evaluation_manifest_hashes={split:sha(HERE/'eval_data'/f'{split}.jsonl') for split in ('development','test')},
        historical_recipe_adaptation='Zapformer has no frozen WavLM frontend; train acoustic backbone and existing head, no new output classes')
    checkpoints=sorted(a.output.glob('step-*.pt'))
    progress=dict(global_step=0,epoch=0,step_in_epoch=0,audio_seconds_seen=0.,save_every_steps=a.save_every)
    if checkpoints:
        progress=step_checkpoint.load(checkpoints[-1],net,optimizer,scheduler,schedule,contract)
    else:
        torch.manual_seed(42+rank);random.seed(42+rank);np.random.seed(42+rank)
    run=None
    if rank==0:
        config=dict(contract=contract,job=os.environ.get('PBS_JOBID'),recorded_at=datetime.now(timezone.utc).isoformat())
        (a.output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
        state_rng=step_checkpoint.rng_state();run=open_wandb(a.output,config,a.smoke);step_checkpoint.restore_rng(state_rng)
    dist.barrier()

    def log(record, prefix=None):
        if rank: return
        with (a.output/'updates.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        if prefix:
            run.log(dict(optimizer_step=progress['global_step'],**{prefix+'/'+k:v for k,v in record.items()
                         if isinstance(v,(int,float))}))
        print(json.dumps(record),flush=True)

    def score(split):
        saved=step_checkpoint.rng_state();training=net.training
        try:
            net.eval()
            result=evaluate(net,split,rank,world,a.output/'evaluations'/f'step-{progress["global_step"]:09d}'/split/f'rank-{rank}',
                            limit=min(world*2,4) if a.smoke else 0)
        finally:
            net.train(training);step_checkpoint.restore_rng(saved)
        if rank==0:
            record=dict(event='timit_evaluation',optimizer_step=progress['global_step'],epoch=progress['epoch'],
                        intermediate_diagnostic=True,test_checkpoint_selection=False,**result)
            with (a.output/'timit_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            (a.output/f'latest_{split}_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
            prefix='timit_dev' if split=='development' else 'timit_test'
            payload={'optimizer_step':progress['global_step'],prefix+'/epoch':progress['epoch']}
            for mode in ('text_free','forced_k2','mapped_arpabet'):
                payload.update({prefix+'/'+mode+'/'+k:v for k,v in result[mode].items() if isinstance(v,(int,float))})
            run.log(payload);run.summary.update(payload)
            print('TIMIT',split,json.dumps({m:result[m] for m in ('text_free','mapped_arpabet')}),flush=True)

    def save():
        step_checkpoint.save(a.output/f'step-{progress["global_step"]:09d}.pt',net,optimizer,scheduler,
                             schedule.state_dict(),progress,contract)
        if rank==0:
            (a.output/'progress.json').write_text(json.dumps(progress)+'\n')
            run.summary.update({'progress/'+k:v for k,v in progress.items()})

    if not checkpoints:
        score('development');score('test')
    dataset=Features(rows)
    for epoch in range(progress['epoch'],a.epochs):
        if schedule.epoch != epoch: schedule.epoch=epoch;schedule.cursor=0
        batches=schedule.batches();remaining=batches[schedule.cursor:]
        loader=DataLoader(dataset,batch_sampler=remaining,num_workers=2,collate_fn=collate,
                          pin_memory=True,generator=torch.Generator().manual_seed(42+epoch+rank))
        for batch_rows,cpu_x,cpu_lengths in loader:
            step_started=time.monotonic()
            gold=targets(batch_rows,cpu_lengths,torch.device('cpu'))
            valid=torch.tensor(int((gold!=-100).sum()),device='cuda',dtype=torch.float64)
            dist.all_reduce(valid);assert valid>0
            optimizer.zero_grad(set_to_none=True)
            for m in net.modules():
                if hasattr(m,'batch_count'):m.batch_count=origin_step+progress['global_step']
            groups=microbatching.plan(cpu_lengths.tolist(),'spottc');total=0.
            for number,indices in enumerate(groups):
                from contextlib import nullcontext
                with model.no_sync() if number<len(groups)-1 else nullcontext():
                    longest=int(cpu_lengths[indices].max())
                    x=cpu_x[indices,:longest].cuda(non_blocking=True)
                    lengths=cpu_lengths[indices].cuda(non_blocking=True)
                    y=gold[indices,:longest].cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=model(x,lengths,y)
                    loss=value*world/valid
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite frame CE')
                    loss.backward();total+=float(value.detach())
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            stats=torch.tensor([total,sum(r['duration'] for r in batch_rows),
                float(a.max_wall_seconds>0 and time.monotonic()-started>=a.max_wall_seconds)],device='cuda',dtype=torch.float64)
            dist.all_reduce(stats)
            progress['global_step']+=1;progress['step_in_epoch']+=1;progress['audio_seconds_seen']+=float(stats[1]);schedule.cursor+=1
            log(dict(event='update',step=progress['global_step'],epoch=epoch+schedule.cursor/len(batches),
                     frame_ce_loss=float(stats[0]/valid),raw_ce_nll_sum=float(stats[0]),valid_frames=int(valid),
                     lr=optimizer.param_groups[0]['lr'],grad_norm_rank0=float(norm),
                     step_seconds=time.monotonic()-step_started,audio_seconds=float(stats[1])), 'train')
            stop=bool(stats[2]) or (a.stop_after>0 and progress['global_step']>=a.stop_after)
            if progress['global_step']%a.save_every==0 or stop: save()
            if progress['global_step']%a.eval_every==0: score('development')
            if stop:
                if a.smoke: score('development');score('test')
                if rank==0:
                    (a.output/('SMOKE_PASS.json' if a.smoke else 'SEGMENT_PAUSED.json')).write_text(json.dumps(progress)+'\n')
                    run.finish()
                dist.destroy_process_group();return
        progress['epoch']=epoch+1;progress['step_in_epoch']=0;schedule.epoch=epoch+1;schedule.cursor=0
        save();score('development');score('test')
    if rank==0:
        artifact=__import__('wandb').Artifact(a.output.name+'-timit-summary',type='evaluation')
        for p in [a.output/'timit_metrics.jsonl',a.output/'run_config.json',HERE/'timeline_provenance.json']:
            artifact.add_file(str(p))
        run.log_artifact(artifact);run.summary['training/complete']=True
        (a.output/'TRAIN_COMPLETE.json').write_text(json.dumps(progress)+'\n');run.finish()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__': main()
