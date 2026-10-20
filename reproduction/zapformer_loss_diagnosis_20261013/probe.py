"""Read-only checkpoint probes and isolated synthetic OTTC convergence tests."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'zapformer_s3t2ss_native_20261012'
ADAPTER = HERE.parent / 'zapformer_libri100_ablation_20261013'
sys.path.insert(0, str(NATIVE))
import runtime
import torch
import train as native
from task_model import S3T2, parameters
from data import Inputs, jsd_rows, CE_ROOT, read_rows
from ifmdd_losses import expand_subphonetic_targets, pad_flat_targets, make_ottc_inputs
from archived_ot_loss import batched_ottc_loss_bucketized
from k2_determinism import enable
sys.path.insert(0, str(ADAPTER))
import task_adapter
from task_adapter import ConfigurableS3T2
from forward_sum import validate_config


def reset():
    torch.manual_seed(42)
    random.seed(42)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def tensors(batch):
    return (batch['inputs'].cuda(), batch['supervisions']['num_frames'].cuda(),
            batch['tokens'].cuda(), batch['token_lengths'].cuda())


def group(name):
    if name.startswith('weight_head'): return 'mass_head'
    if name.startswith('acoustic.ctc_output'): return 'state_head'
    if name.startswith('acoustic.encoder_embed'): return 'frontend'
    return 'encoder'


def gradients(net, batch, aux):
    reset(); net.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, components = net(*tensors(batch), aux_loss_scale=aux)
    loss.backward()
    grads = {name: p.grad.detach().clone() for name,p in net.named_parameters() if p.grad is not None}
    values = {k: float(v) for k,v in components.items()}
    values['total'] = float(loss.detach())
    return grads, values


def compare_gradients(base, full):
    result = {}
    for g in ['frontend','encoder','state_head','mass_head']:
        names = [n for n in base if group(n)==g]
        nb = sum(float(base[n].double().square().sum()) for n in names)
        nf = sum(float(full[n].double().square().sum()) for n in names)
        nd = sum(float((full[n]-base[n]).double().square().sum()) for n in names)
        dot = sum(float((base[n].double()*(full[n]-base[n]).double()).sum()) for n in names)
        result[g] = dict(base_gradient_norm=math.sqrt(nb), actual_gradient_norm=math.sqrt(nf),
                         auxiliary_delta_norm=math.sqrt(nd), delta_to_base_ratio=math.sqrt(nd/max(nb,1e-30)),
                         base_delta_cosine=dot/math.sqrt(max(nb*nd,1e-30)),
                         parameters_with_gradient=len(names))
    return result


def output_stats(net, batch):
    capture = {}
    original = task_adapter.batched_ottc_loss_bucketized
    def observed(x,y,a,b,**kwargs):
        result = original(x,y,a,b,**kwargs)
        capture.update(x=x,y=y,a=a,b=b,amask=kwargs['amask'],bmask=kwargs['bmask'],result=result)
        return result
    task_adapter.batched_ottc_loss_bucketized = observed
    net.eval();reset()
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss,components=net(*tensors(batch),aux_loss_scale=0.)
        x,y,a,b=capture['x'],capture['y'],capture['a'],capture['b']
        mask=capture['amask'];p=x.softmax(-1)
        _,ix,iy,h=capture['result'];ix=ix.long();iy=iy.long()
        yp=y.argmax(-1).gather(1,iy)
        xp=p.gather(1,ix[...,None].expand(-1,-1,p.shape[-1])).gather(-1,yp[...,None]).squeeze(-1)
        entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
        masses=[]
        for i in range(x.shape[0]):
            n=int(mask[i].sum());w=a[i,:n]
            masses.append(dict(max_mass=float(w.max()),effective_frame_fraction=float((-(w*w.clamp_min(1e-30).log()).sum()).exp()/n)))
        effective_ids=torch.cat([torch.zeros(1,device='cuda',dtype=torch.long),torch.arange(7,313,device='cuda')])
        raw=effective_ids[p.argmax(-1)];state=raw[mask]
        phones=torch.where(state>0,(state-1)//3+1,0)
        hist=torch.bincount(phones,minlength=105)
        phases=torch.bincount(((state[state>0]-1)%3),minlength=3)
        return dict(loss=float(loss),components={k:float(v) for k,v in components.items()},
                    utterances=x.shape[0],frames=int(mask.sum()),
                    class_entropy_mean=float(entropy[mask].mean()),uniform_class_entropy=math.log(p.shape[-1]),
                    posterior_max_mean=float(p.max(-1).values[mask].mean()),
                    transport_target_probability=float((xp*h).sum()/x.shape[0]),
                    total_transport_mass_per_utterance=(h.sum(-1)).tolist(),
                    target_mass_sums=b.sum(-1).tolist(),target_onehot_sums=y.sum(-1)[capture['bmask']].unique().tolist(),
                    blank_argmax_fraction=float((state==0).float().mean()),
                    distinct_argmax_phones=int((hist>0).sum()),argmax_state_phase_counts=phases.tolist(),
                    dominant_phones=[dict(phone=int(i),fraction=float(hist[i]/hist.sum())) for i in hist.topk(5).indices],
                    mass_statistics=masses)
    finally:task_adapter.batched_ottc_loss_bucketized=original


def toy(run):
    results={}
    for states in [1,3]:
        for jsd in [True,False]:
            reset(); t=60;c=1+104*states
            labels=torch.tensor([3,4,3],device='cuda');length=torch.tensor([3],device='cuda')
            expanded,el=expand_subphonetic_targets(labels,length,states)
            target=pad_flat_targets(expanded,el)
            logits=torch.nn.Parameter(torch.randn(1,t,c,device='cuda')*.02)
            mass=torch.nn.Parameter(torch.zeros(1,t,device='cuda'))
            opt=torch.optim.Adam([logits,mass],lr=.15)
            ids=torch.cat([torch.zeros(1,device='cuda',dtype=torch.long),torch.arange(1+2*states,c,device='cuda')])
            history=[]
            for step in range(201):
                a,b,fm,y=make_ottc_inputs(logits,torch.tensor([t],device='cuda'),target,el,mass)
                tm=torch.arange(target.shape[1],device='cuda')[None]<el[:,None]
                loss=batched_ottc_loss_bucketized(logits.index_select(-1,ids),y.index_select(-1,ids),a,b,amask=fm,bmask=tm,jsd=jsd)[0]
                if step%20==0:
                    row=dict(step=step,loss=float(loss.detach()));history.append(row)
                    run.log({'toy/'+f'states{states}_'+('jsd' if jsd else 'ce')+'/loss':row['loss'],'diagnostic_step':step})
                if step==200:break
                opt.zero_grad();loss.backward();opt.step()
            results[f'states{states}_'+('jsd' if jsd else 'ce')]=dict(history=history,
                optimizer='Adam .15 for synthetic free logits/masses only; not a Zapformer recipe',
                target_phones=labels.tolist(),target_state_ids=expanded.tolist())
            print(json.dumps({'toy':list(results)[-1],'initial':history[0]['loss'],'final':history[-1]['loss']}),flush=True)
    return results


def main():
    from reproduction.wandb_policy import init_evaluation
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;enable()
    import wandb
    defaults=json.loads((HERE.parent/'wandb_config.json').read_text())
    identity=HERE/'wandb_run.json'
    saved=json.loads(identity.read_text()) if identity.exists() else {}
    opts=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
              name='zapformer_loss_diagnosis_20261013',group='zapformer_diagnostics',dir=str(HERE),
              id=saved.get('id'),resume='allow' if saved else None,mode=os.environ.get('WANDB_MODE','online'),
              config=dict(purpose='read-only five-checkpoint gradient/output probes and isolated synthetic SP convergence',seed=42,
                          source_sha256=sha(__file__),formal_training_changed=False))
    try:run=init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode']='offline';opts.pop('resume',None);run=init_evaluation(**opts)
    identity.write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode))+'\n')
    rows=jsd_rows()
    train_rows=next(iter([[r for r in rows if 350<=r['frames']<=550 and len(r['tokens'])>=8][:4]]))
    dev_rows=read_rows(CE_ROOT/'eval_data/development.jsonl')[:4]
    for row in dev_rows:row['intervals']=[[p['mfa_id'],p['start'],p['end']] for p in row['phone_intervals']]
    batches=dict(train=Inputs(train_rows).batch([r['id'] for r in train_rows]),
                 development=Inputs(dev_rows).batch([r['id'] for r in dev_rows]))
    report=dict(gpu=torch.cuda.get_device_name(),source_sha256=sha(__file__),probe_inputs={k:[r['id'] for r in b['rows']] for k,b in batches.items()},
                gradient_protocol='Fixed four-utterance Libri100 batch; native train mode BF16, same RNG for both passes, no SpecAugment; unscaled gradients. Baseline retains orthogonal penalties, delta measures aux_loss_scale-controlled limiters only.',
                checkpoints={},toy_convergence={})
    for name,info in json.loads((HERE/'inputs.json').read_text()).items():
        reset();state=torch.load(info['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        contract=state['contract'];step=state['progress']['global_step']
        config=validate_config(contract.get('continuation',{}).get('loss_config',dict(ottc_cost='jsd_upstream',forward_sum_weight=0.)))
        # Origin's posterior/transport probe uses OTTC-only; its unchanged full
        # native objective is used separately for the gradient comparison.
        net=ConfigurableS3T2('jsd',topology='s3_t2_star_star',loss_config=config)
        net.load_state_dict(state['model'],strict=True);net.cuda()
        result=dict(step=step,epoch=state['progress']['epoch'],checkpoint_sha256=sha(info['checkpoint']),loss_config=config,
                    original_native_objective_for_origin_gradients=name=='origin',outputs={})
        for split,batch in batches.items():result['outputs'][split]=output_stats(net,batch)
        if name=='origin':
            native_net=S3T2('jsd',topology='s3_t2_star_star');native_net.load_state_dict(state['model'],strict=True)
            del net;torch.cuda.empty_cache();net=native_net.cuda()
        del state
        net.train();params=parameters();params.update(batch_idx_train=step,max_duration=640,world_size=8)
        native.set_batch_count(net,native.get_adjusted_batch_count(params))
        base,base_values=gradients(net,batches['train'],0.)
        aux=.05*(.25 if step>2000 else 1.)
        full,full_values=gradients(net,batches['train'],aux)
        assert base_values==full_values,'Auxiliary scale unexpectedly changed scalar forward loss'
        result['gradients']=compare_gradients(base,full)
        result['aux_loss_scale_unscaled']=aux;result['gradient_probe_loss']=base_values
        del base,full,net;torch.cuda.empty_cache()
        report['checkpoints'][name]=result
        (HERE/'gpu_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'checkpoint':name,'step':step,'gradients':result['gradients']}),flush=True)
        for g,metrics in result['gradients'].items():
            run.log({f'probe/{name}/{g}/{k}':v for k,v in metrics.items()})
    report['toy_convergence']=toy(run)
    (HERE/'gpu_report.json').write_text(json.dumps(report,indent=2)+'\n')
    artifact=wandb.Artifact('zapformer-loss-diagnosis-20261013',type='diagnostic')
    for f in ['gpu_report.json','inputs.json','probe.py']:artifact.add_file(str(HERE/f))
    run.log_artifact(artifact);run.finish()
    (HERE/'GPU_PROBE_COMPLETE.json').write_text(json.dumps(dict(source_sha256=sha(__file__),checkpoint_count=len(report['checkpoints']),toy_cases=len(report['toy_convergence'])))+'\n')


if __name__=='__main__':main()
