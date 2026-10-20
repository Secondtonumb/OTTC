"""Recipe smoke: real features, active auxiliary gradients and exact resume."""
import runtime
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model_s3t2 import MFAZapformer
from batched_rubik import BatchedRubik
from icefall.utils import get_parameter_groups_with_lrs
from icefall.checkpoint import update_averaged_model
from variable_combined_scheduler import InterpCosineLRScheduler
from alternating_spec_augment import AlternatingSpecAugment
from feature_manifest import load_split
from lhotse_batches_scaling import make_sampler
import microbatching
import checkpoint

HERE=Path(__file__).resolve().parent


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
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true')
    ap.add_argument('--resume-phase',action='store_true');ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--ddp',action='store_true')
    a=ap.parse_args();device='cpu' if a.cpu else 'cuda'
    rank=int(os.environ.get('RANK',0)) if a.ddp else 0
    world=int(os.environ.get('WORLD_SIZE',1)) if a.ddp else 1
    if a.ddp:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    shared=HERE/'validation'/('ddp' if a.ddp else 'cpu' if a.cpu else 'full' if a.full else 'tiny')
    out=shared/('rank'+str(rank)) if a.ddp else shared;out.mkdir(parents=True,exist_ok=True)
    contract=dict(validation=True,full=a.full,cpu=a.cpu,world=world,recipe='zapformer_archived_tricks_v1',
                  cpu_scope='backbone surrogate only; archived JSD requires CUDA' if a.cpu else 'actual JSD+S3T2')
    net=MFAZapformer(tiny=not a.full).to(device)
    avg=copy.deepcopy(net).cpu().double() if not rank else None
    forward_net=DDP(net,device_ids=[int(os.environ['LOCAL_RANK'])],broadcast_buffers=False) if a.ddp else net
    optimizer=BatchedRubik(get_parameter_groups_with_lrs(net,lr=.02,include_names=False),lr=.02,beta1=.99)
    scheduler=InterpCosineLRScheduler(optimizer,batches_per_epoch=[6887]*10,min_factor=.025,linear_scale=.5)
    scheduler.set_epoch(1);scheduler.set_batch(0)
    scaler=torch.amp.GradScaler('cuda',enabled=not a.cpu,init_scale=1.)
    aug=AlternatingSpecAugment(seed=92+rank)
    # CPU validates all state transitions with an inexpensive topology of the same model.
    if a.cpu:
        rows=[dict(id='test'+str(i),frames=80+i,duration=(80+i)/100.,tokens=[3,4,5]) for i in range(6)]
        features=[torch.randn(r['frames'],80) for r in rows]
    else:
        meta,all_rows=load_split(Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json'))
        sampler0=make_sampler(all_rows,320.,rank,30.,world);sampler0.set_epoch(0)
        ids=[cut.id for cut in next(iter(sampler0))];lookup={r['id']:r for r in all_rows}
        rows=[lookup[i] for i in ids]
        # Full model exercises a complete real per-rank batch and its actual microbatch plan.
        if not a.full:rows=rows[:6]
        features=[]
        for r in rows:
            path=Path(r.get('feature_file') or meta['feature_file'])
            if not path.is_absolute():path=Path('/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2')/path
            mmap=np.memmap(path,mode='r',dtype=np.float32)
            features.append(torch.from_numpy(np.array(mmap[r['feature_offset']:r['feature_offset']+80*r['frames']]).reshape(-1,80)))
    sampler_rows=[dict(r,id=r['id']+'_'+str(i)) for i in range(6) for r in rows] if len(rows)<30 else rows
    sampler=make_sampler(sampler_rows,1.,0,30.,1);sampler.set_epoch(0)
    next(iter(sampler))
    cpu=(torch.nn.utils.rnn.pad_sequence(features,batch_first=True),torch.tensor([len(x) for x in features]),
         torch.tensor([p for r in rows for p in r['tokens']]),torch.tensor([len(r['tokens']) for r in rows]))
    microbatching.LIMITS['spottc']=(28000,40000000)
    def update(step):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device,enabled=False):x=aug(cpu[0].to(device).float(),cpu[1].to(device))
        value_sum=0.
        for indices in microbatching.plan(cpu[1].tolist(),'spottc'):
            b=tuple(t.to(device) for t in microbatching.tensors(cpu,indices))
            b=(x[indices[0]:indices[-1]+1,:b[0].shape[1]],*b[1:])
            with torch.autocast(device,dtype=torch.bfloat16,enabled=not a.cpu):
                if a.cpu:
                    # Archived JSD explicitly requires CUDA. CPU checks the shared
                    # backbone/optimizer/augmentation state machine; GPU checks JSD.
                    enc,_=net.acoustic.forward_encoder(b[0],b[1],aux_loss_scale=.05)
                    logits=net.acoustic.ctc_output(enc)
                    value=(logits.square().sum()+net.weight_head(enc).square().sum(),)
                else:
                    value=forward_net(*b,aux_loss_scale=scaler.get_scale()*.05)
            assert bool(torch.isfinite(value[0]))
            scaler.scale(value[0]).backward();value_sum+=float(value[0].detach())
        scaler.unscale_(optimizer)
        assert all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
        assert net.weight_head.weight.grad is not None
        scheduler.set_batch(step);scaler.step(optimizer);scaler.update()
        # Test the exact reference averaging operation without waiting200 updates.
        params=type('P',(),dict(average_period=1,batch_idx_train=step+1))()
        if avg is not None:update_averaged_model(params,net,avg)
        if a.ddp:
            values=[None]*world;dist.all_gather_object(values,digest(net.state_dict()))
            assert len(set(values))==1,'DDP models diverged'
        return value_sum
    def state_digest():
        return digest(dict(model=net.state_dict(),avg=avg.state_dict() if avg is not None else None,
                           optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                           sampler=sampler.state_dict(),random=checkpoint.random_state(aug)))
    path=shared/'checkpoint.pt'
    if a.resume_phase:
        progress=checkpoint.load(path,net,avg,optimizer,scheduler,scaler,aug,sampler,contract)
        restored=state_digest()
        expected=json.loads((out/'expected.json').read_text())
        assert restored==expected['checkpoint_digest'],'Checkpoint restore mismatch'
        value=update(1)
        actual=state_digest()
        assert actual==expected['continued_digest'],'Continued RNG/model/optimizer update mismatch'
        result=dict(status='PASS',full=a.full,cpu=a.cpu,world=world,fresh_process_exact_resume=True,loss=value,
                    optimizer='BatchedRubik',aux_loss_scale=.05,private_specaug_rng_saved=True,
                    scheduler='InterpCosineLRScheduler',model_avg_saved=True,samples=len(rows),
                    microbatches=len(microbatching.plan(cpu[1].tolist(),'spottc')),
                    peak_GiB=torch.cuda.max_memory_allocated()/2**30 if not a.cpu else None)
        result['objective_scope']='backbone surrogate only' if a.cpu else 'actual archived upstream JSD + S3T2'
        (out/'SMOKE_PASS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if a.ddp:
            dist.barrier()
            if not rank:(shared/'SMOKE_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        value=update(0)
        progress=dict(global_step=1,epoch=0,step_in_epoch=1)
        checkpoint.save(path,net,avg,optimizer,scheduler,scaler,aug,sampler.state_dict(),progress,contract)
        restored=state_digest()
        update(1)
        continued=state_digest()
        (out/'expected.json').write_text(json.dumps(dict(checkpoint_digest=restored,continued_digest=continued))+'\n')
        print('Saved resume oracle, first loss '+str(value),flush=True)
    if a.ddp:dist.destroy_process_group()


if __name__=='__main__':main()
