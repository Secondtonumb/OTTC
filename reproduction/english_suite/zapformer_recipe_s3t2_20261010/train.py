"""Independent MFA SPOTTC-JSD: archived Zapformer recipe training mechanisms.

Keeps the 10ms phone task adaptation, enabling recipe LR decay, auxiliary
gradients, AlternatingSpecAugment, AMP scaling and cumulative averaging.
"""
import runtime
import argparse,copy,hashlib,json,math,os,random,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model_s3t2 import MFAZapformer,parameters
from batched_rubik import BatchedRubik
from icefall.utils import get_parameter_groups_with_lrs
from icefall.checkpoint import update_averaged_model
from variable_combined_scheduler import InterpCosineLRScheduler
from alternating_spec_augment import AlternatingSpecAugment
from lhotse_batches_scaling import make_sampler
import checkpoint as recipe_checkpoint
import microbatching
from contextlib import nullcontext
from collections import deque
from feature_manifest import load_split
from tracking import Tracker

BASE=Path(__file__).resolve().parent

def digest(value):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',default='spottc',choices=['spottc'])
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-duration',type=float,required=True);ap.add_argument('--num-epochs',type=int,default=10)
    ap.add_argument('--lr',type=float,default=.02);ap.add_argument('--warmup-steps',type=int,default=0)
    ap.add_argument('--saves-per-epoch',type=int,default=4);ap.add_argument('--save-every-steps',type=int,default=0)
    ap.add_argument('--stop-after-step',type=int,default=0);ap.add_argument('--resume',type=Path)
    ap.add_argument('--allow-pilot',action='store_true');ap.add_argument('--tiny',action='store_true')
    ap.add_argument('--migration-contract',type=Path)
    ap.add_argument('--loader-workers',type=int,default=4)
    ap.add_argument('--prefetch-batches',type=int,default=8)
    ap.add_argument('--max-wall-seconds',type=float,default=0)
    ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    if not 1<=a.loader_workers<=a.prefetch_batches:raise ValueError('Require 1 <= loader-workers <= prefetch-batches')
    run_started=time.monotonic()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl');assert world==8
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    meta,rows=load_split(a.data)
    if not meta.get('training_ready',False) and not a.allow_pilot:
        raise ValueError('Full training manifest is not ready; --allow-pilot is only for explicit pilot validation')
    if rank==0:print("Metadata loaded",len(rows),flush=True)
    lookup={r['id']:r for r in rows};assert len(lookup)==len(rows)
    net=MFAZapformer(tiny=a.tiny).cuda()
    model=DDP(net,device_ids=[local],broadcast_buffers=False);p=parameters(a.tiny);p.base_lr=a.lr
    optimizer=BatchedRubik(get_parameter_groups_with_lrs(net,lr=a.lr,include_names=False),lr=a.lr,beta1=.99)
    model_avg=copy.deepcopy(net).cpu().double() if rank==0 else None
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=1.)
    specaug=AlternatingSpecAugment(seed=42)
    # Fixed split limits are part of this independent experiment's resume contract.
    microbatching.LIMITS['spottc']=(28000,40000000)
    # Estimate once from native sampler metadata, then keep the save interval fixed.
    estimate=[sum(1 for _ in make_sampler(rows,a.max_duration,0,30.,world)) if rank==0 else None]
    dist.broadcast_object_list(estimate,src=0);epoch_steps=estimate[0]
    if rank==0:print("Epoch schedule",epoch_steps,flush=True)
    interval=a.save_every_steps or max(1,math.ceil(epoch_steps/a.saves_per_epoch))
    if interval<=0:raise ValueError('save interval must be positive')
    if a.warmup_steps or a.migration_contract:
        raise ValueError('This recipe has its own LR schedule and cannot migrate older experiment state')
    scheduler=InterpCosineLRScheduler(optimizer,batches_per_epoch=[epoch_steps]*a.num_epochs,
                                    min_factor=.025,linear_scale=.5)
    scheduler.set_epoch(1);scheduler.set_batch(0)
    sampler=make_sampler(rows,a.max_duration,rank,30.,world)
    contract=dict(arm=a.arm,tiny=a.tiny,manifest_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        num_epochs=a.num_epochs,max_duration=a.max_duration,quadratic_duration=30,num_buckets=30,lr=a.lr,
        warmup_steps=a.warmup_steps,save_every_steps=interval,estimated_epoch_steps=epoch_steps,
        tokens_sha256=hashlib.sha256(Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/outputs/mfa_corpus_audit_20260921/training_inventory/tokens.txt').read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(BASE.rglob('*.py')) if 'training' not in path.relative_to(BASE).parts and 'validation' not in path.relative_to(BASE).parts})
    contract.update(recipe='zapformer_archived_tricks_v1',ottc_cost='jsd_upstream',topology='s3_t2',
                    frame_shift_ms=10,initialization='seed42 random',optimizer='BatchedRubik',beta1=.99,
                    loss_gradient_reduction='local batch sum, averaged across DDP ranks',
                    scheduler=dict(name='InterpCosineLRScheduler',min_factor=.025,linear_scale=.5),
                    auxiliary=dict(base_scale=.05,after_step2000_factor=.25,coupled_to_grad_scaler=True),
                    specaug='archived AlternatingSpecAugment defaults, per-rank/epoch private RNG checkpointed',
                    average_period=200,gradient_clipping=None,microbatch_limits=microbatching.LIMITS['spottc'],
                    num_copies=1,output_initial_scale=.1,
                    adaptation='MFA features/phone targets and 10ms grid retained; no new speed perturbation manifest')
    progress=dict(global_step=0,epoch=0,step_in_epoch=0,save_every_steps=interval,audio_seconds_seen=0.,
                  optimizer_updates=0,averaged_at_step=0,recipe_epoch=0)
    torch.manual_seed(42+rank);random.seed(42+rank);np.random.seed(42+rank)
    if a.resume:
        progress=recipe_checkpoint.load(a.resume,net,model_avg,optimizer,scheduler,scaler,specaug,sampler,contract)
        if rank==0:print("Resumed",progress,flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        (a.output/'run_config.json').write_text(json.dumps(dict(world_size=world,contract=contract,resume=str(a.resume),loader_workers=a.loader_workers,prefetch_batches=a.prefetch_batches),indent=2)+'\n')
        if a.audit:
            (a.output/'loaded_state.json').write_text(json.dumps(dict(progress=progress,model_digest=digest(net.state_dict()),optimizer_digest=digest(optimizer.state_dict()),scheduler=scheduler.state_dict()),indent=2)+'\n')
    caches={}
    tracker=Tracker(a.output,contract,rank)
    def load(cuts):
        # CUDA current-device state is thread-local, including pinned allocation.
        torch.cuda.set_device(local)
        load_started=time.perf_counter()
        selected=[lookup[c.id] for c in cuts];xs=[]
        for r in selected:
            f=Path(r.get('feature_file') or meta['feature_file'])
            if not f.is_absolute():f=a.data.parent/f
            if str(f) not in caches:caches[str(f)]=np.memmap(f,mode='r',dtype=np.float32)
            x=caches[str(f)][r['feature_offset']:r['feature_offset']+80*r['frames']]
            xs.append(torch.from_numpy(np.array(x).reshape(-1,80)))
        return selected,(torch.nn.utils.rnn.pad_sequence(xs,batch_first=True).pin_memory(),
            torch.tensor([r['frames'] for r in selected]).pin_memory(),
            torch.tensor([v for r in selected for v in r['tokens']]).pin_memory(),
            torch.tensor([len(r['tokens']) for r in selected]).pin_memory()),time.perf_counter()-load_started
    def log(record):
        if rank==0:
            with (a.output/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        tracker.log(record)
    def checkpoint(consumed):
        start=time.perf_counter()
        if a.audit:
            if rank==0:log(dict(event='checkpoint_state',step=progress['global_step'],model_digest=digest(net.state_dict()),optimizer_digest=digest(optimizer.state_dict()),scheduler=scheduler.state_dict()))
        path=a.output/('step-{:09d}.pt'.format(progress['global_step']))
        recipe_checkpoint.save(path,net,model_avg,optimizer,scheduler,scaler,specaug,consumed,progress,contract)
        log(dict(event='checkpoint',step=progress['global_step'],seconds=time.perf_counter()-start,path=str(path)))
        if rank==0:(a.output/'progress.json').write_text(json.dumps(progress)+'\n')
    last_saved=-1
    initial_step=progress['global_step']
    tracker.development(net,progress)
    for epoch in range(progress['epoch'],a.num_epochs):
        net.train()
        if progress['recipe_epoch']!=epoch+1:
            # Shared sampler seeds, independent augmentation/CUDA seeds per rank/epoch.
            random.seed(42+epoch);np.random.seed(42+epoch)
            seed=42+50*(epoch+1)+512*rank
            torch.cuda.manual_seed(seed)
            specaug=AlternatingSpecAugment(seed=seed)
            progress['recipe_epoch']=epoch+1
        scheduler.set_epoch(epoch+1)
        sampler.set_epoch(epoch);iterator=iter(sampler)
        with ThreadPoolExecutor(max_workers=a.loader_workers) as pool:
            def enqueue():
                try:cuts=next(iterator)
                except StopIteration:return None
                # Capture position at THIS batch, before prefetching the next one.
                return copy.deepcopy(sampler.state_dict()),pool.submit(load,cuts)
            pending=deque()
            for _ in range(a.prefetch_batches):
                item=enqueue()
                if item is None:break
                pending.append(item)
            while pending:
                step_started=time.perf_counter()
                consumed,future=pending.popleft();selected,cpu,load_seconds=future.result()
                input_wait=time.perf_counter()-step_started
                sample_started=time.perf_counter();item=enqueue()
                if item is not None:pending.append(item)
                sample_seconds=time.perf_counter()-sample_started
                count=torch.tensor(len(selected),device='cuda',dtype=torch.float32);dist.all_reduce(count)
                for m in net.modules():
                    if hasattr(m,'batch_count'):m.batch_count=progress['global_step']
                optimizer.zero_grad(set_to_none=True)
                groups=microbatching.plan([r['frames'] for r in selected],a.arm)
                local_total=0.;local_jsd=0.;local_topology=0.
                # Augment the complete per-rank logical batch before partitioning it.
                with torch.autocast('cuda',enabled=False):
                    augmented=specaug(cpu[0].cuda(non_blocking=True).float(),cpu[1].cuda(non_blocking=True))
                current_scale=scaler.get_scale()
                aux_scale=current_scale*.05*(.25 if progress['global_step']+1>2000 else 1.)
                torch.cuda.reset_peak_memory_stats()
                for index,indices in enumerate(groups):
                    context=nullcontext() if index==len(groups)-1 else model.no_sync()
                    with context:
                        b=tuple(t.cuda(non_blocking=True) for t in microbatching.tensors(cpu,indices))
                        b=(augmented[indices[0]:indices[-1]+1,:b[0].shape[1]],*b[1:])
                        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(*b,aux_loss_scale=aux_scale)
                        local_sum=value[0]
                        loss=local_sum
                        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
                        scaler.scale(loss).backward();local_total+=float(local_sum.detach())
                        local_jsd+=float(value[3]);local_topology+=float(value[4])
                        del b,value,local_sum,loss
                del augmented
                scaler.unscale_(optimizer)
                norm=torch.stack([v.grad.detach().float().square().sum() for v in net.parameters() if v.grad is not None]).sum().sqrt()
                scheduler.set_batch(progress['step_in_epoch'])
                finite=bool(torch.isfinite(norm))
                scaler.step(optimizer);scaler.update()
                if finite:progress['optimizer_updates']+=1
                new_scale=scaler.get_scale();batch_index=progress['step_in_epoch']
                if new_scale<1e-5:raise FloatingPointError('AMP scale too small')
                if (batch_index%25==0 and new_scale<2 or batch_index%100==0 and new_scale<8 or batch_index%400==0 and new_scale<32):
                    scaler.update(new_scale*2)
                memory=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device='cuda',dtype=torch.int64)
                dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                stats=torch.tensor([local_total,sum(r['duration'] for r in selected),float(a.max_wall_seconds>0 and time.monotonic()-run_started>=a.max_wall_seconds)],device='cuda',dtype=torch.float64);dist.all_reduce(stats)
                progress['global_step']+=1;progress['step_in_epoch']+=1;progress['audio_seconds_seen']+=float(stats[1])
                if rank==0 and progress['global_step']%200==0:
                    average_params=type('AverageParams',(),dict(average_period=200,batch_idx_train=progress['global_step']))()
                    update_averaged_model(average_params,net,model_avg)
                if progress['global_step']%200==0:progress['averaged_at_step']=progress['global_step']
                timing=torch.tensor([input_wait,load_seconds,sample_seconds],device='cuda',dtype=torch.float64)
                dist.all_reduce(timing,op=dist.ReduceOp.MAX)
                ids=[r['id'] for r in selected];all_ids=[None]*world if rank==0 else None
                if a.audit:dist.gather_object(ids,all_ids,dst=0)
                components=torch.tensor([local_jsd,local_topology,float(sum(r['frames'] for r in selected))],device='cuda',dtype=torch.float64);dist.all_reduce(components)
                log(dict(event='update',step=progress['global_step'],epoch=epoch,loss=float(stats[0]),loss_reduction='sum_all_ranks',loss_per_utterance=float(stats[0]/count),loss_per_frame=float(stats[0]/components[2]),jsd_sum=float(components[0]),topology_sum=float(components[1]),lr=scheduler.get_last_lr()[0],rank_ids=all_ids if a.audit else None,microbatches_rank0=len(groups),step_seconds=time.perf_counter()-step_started,audio_seconds=float(stats[1]),input_wait_max=float(timing[0]),cpu_load_seconds_max=float(timing[1]),sampler_seconds_max=float(timing[2]),peak_allocated_rank0=torch.cuda.max_memory_allocated(),peak_reserved_rank0=torch.cuda.max_memory_reserved(),peak_allocated_max=int(memory[0]),peak_reserved_max=int(memory[1]),grad_norm_rank0=float(norm),grad_scale=current_scale,aux_loss_scale=aux_scale,optimizer_updates=progress['optimizer_updates'],averaged_at_step=progress['averaged_at_step']))
                wall_stop=bool(stats[2]>0)
                del cpu,stats,timing
                due=progress['global_step']%interval==0 or progress['global_step']==initial_step+100
                stop=wall_stop or (a.stop_after_step>0 and progress['global_step']>=a.stop_after_step)
                if due or stop:checkpoint(consumed);last_saved=progress['global_step']
                if stop:
                    tracker.finish();dist.destroy_process_group();return
        progress['epoch']=epoch+1;progress['step_in_epoch']=0
        # An epoch-end fallback is additional protection; periodic saves do not wait for it.
        # Persist the completed-epoch cursor even if a periodic save used this step.
        checkpoint(sampler.state_dict());last_saved=progress['global_step']
        tracker.development(net,progress)
    tracker.finish()
    dist.destroy_process_group()

if __name__=='__main__':main()
