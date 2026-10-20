"""Real cached Libri100 batches; full model, augmentation, BF16 backward and native optimizer."""
import argparse, gc, hashlib, json, math, statistics, time
import torch
from task_ottc import HERE, TopologyOTTC, native
from data import jsd_rows, Inputs, plan
from augmentation import augment
from k2_determinism import enable
from train_ottc import Tracking


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--duration',type=float,required=True);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.autograd.set_multithreading_enabled(False);enable()
    cfg=json.loads((HERE/'configs'/f'{a.arm}.json').read_text())
    rows=jsd_rows();inputs=Inputs(rows);batches=plan(rows,a.duration,0,8,0)
    ordered=sorted(range(len(batches)),key=lambda i:max(inputs.rows[k]['frames'] for k in batches[i]))
    selected=[ordered[0],ordered[len(ordered)//2],ordered[-1],max(range(len(batches)),key=lambda i:len(batches[i])*max(inputs.rows[k]['frames'] for k in batches[i]))]
    out=HERE/'throughput'/a.arm/str(int(a.duration));out.mkdir(parents=True,exist_ok=True)
    report=dict(arm=a.arm,max_duration=a.duration,world_size=1,batch_planning_world_size=8,model='full native Zapformer base',precision='bf16',augmentation='native AlternatingSpecAugment',encoder_aux_loss_scale=.05,optimizer='native BatchedRubik .02 beta1 .99',warmup_steps_per_shape=2,measured_steps_per_shape=3,cached_input=True,omits='DDP communication, checkpoint/evaluation and native parameter plotting',GPU=torch.cuda.get_device_name(),total_memory_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},loss_config=cfg)
    tracking=Tracking(out,report,0);results=[];step=0
    for shape,index in zip(('short','median','longest','largest_padded_batch'),selected):
        torch.manual_seed(42);torch.cuda.manual_seed(92)
        net=TopologyOTTC(loss_config=cfg).cuda();net.train()
        groups=native.get_parameter_groups_with_lrs(net,lr=.02,include_names=False)
        groups=[dict(g,params=[p for p in g['params'] if p.requires_grad]) for g in groups]
        opt=native.BatchedRubik([g for g in groups if g['params']],lr=.02,beta1=.99)
        scaler=native.GradScaler(enabled=True,init_scale=1.)
        specaug=native.AlternatingSpecAugment(seed=92)
        batch=inputs.batch(batches[index]);lengths=batch['supervisions']['num_frames'].cuda();tokens=batch['tokens'].cuda();tl=batch['token_lengths'].cuda()
        native.set_batch_count(net,0);torch.cuda.reset_peak_memory_stats();samples=[]
        for local in range(5):
            torch.cuda.synchronize();started=time.monotonic()
            x=batch['inputs'].cuda()
            with torch.autocast('cuda',enabled=False):x,_=augment(specaug,x.float(),lengths,None)
            scale=1. if scaler._scale is None else scaler.get_scale()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,comp=net(x,lengths,tokens,tl,aux_loss_scale=scale*.05)
            assert math.isfinite(float(loss))
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();opt.zero_grad()
            torch.cuda.synchronize();seconds=time.monotonic()-started;step+=1
            record=dict(optimizer_step=step,epoch=0,shape=shape,warmup=local<2,seconds=seconds,loss=float(loss),lr=max(g['lr'] for g in opt.param_groups),audio_seconds=float(lengths.sum())/100,utterances=len(lengths),max_frames=int(lengths.max()),padding_fraction=1-float(lengths.sum())/batch['inputs'].shape[0]/batch['inputs'].shape[1],peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,**{k:float(v) for k,v in comp.items()})
            record['audio_seconds_per_second']=record['audio_seconds']/seconds;tracking.log(record)
            if local>=2:samples.append(record)
        results.append(dict(shape=shape,batch_index=index,utterance_ids=batches[index],median_seconds=statistics.median(r['seconds'] for r in samples),audio_seconds_per_second=statistics.median(r['audio_seconds_per_second'] for r in samples),peak_reserved_GiB=max(r['peak_reserved_GiB'] for r in samples),peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in samples),padding_fraction=samples[0]['padding_fraction'],audio_seconds=samples[0]['audio_seconds'],max_frames=samples[0]['max_frames']))
        del loss,comp,net,opt,groups,x,scaler,specaug,batch,lengths,tokens,tl;gc.collect();torch.cuda.empty_cache()
    report.update(status='passed',results=results)
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    tracking.run.summary.update(dict(throughput_complete=True));tracking.finish(True)

if __name__=='__main__':
    try:main()
    except torch.cuda.OutOfMemoryError as exc:
        import sys
        arm=sys.argv[sys.argv.index('--arm')+1];duration=sys.argv[sys.argv.index('--duration')+1]
        out=HERE/'throughput'/arm/str(int(float(duration)));out.mkdir(parents=True,exist_ok=True)
        report=dict(status='out_of_memory',error=str(exc),arm=arm,max_duration=float(duration),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
        (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
