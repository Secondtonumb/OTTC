"""Isolated original-OTTC LR x Charsiu forward-sum diagnostics; no formal-run edits."""
import gc
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'zapformer_loss_diagnosis_20261013'
sys.path.insert(0, str(PRIOR))
from probe import (torch, native, parameters, ConfigurableS3T2, validate_config,
                   Inputs, jsd_rows, reset, tensors, sha, enable, output_stats, group)
import task_adapter
from checkpoint import tree_to

BASE_LR = .017118957847577866
STEPS = 120


def new_model(state, weight):
    reset()
    config = validate_config(dict(ottc_cost='jsd_upstream', forward_sum_weight=weight))
    net = ConfigurableS3T2('jsd', topology='s3_t2_star_star', loss_config=config).cuda()
    net.load_state_dict(state['model'], strict=True)
    return net


def set_count(net, step):
    p = parameters()
    p.update(batch_idx_train=step, max_duration=640, world_size=8)
    native.set_batch_count(net, native.get_adjusted_batch_count(p))


def restored_optimizer(net, state, lr):
    opt = native.BatchedRubik(native.get_parameter_groups_with_lrs(net, lr=.02,
                              include_names=False), lr=.02, beta1=.99)
    opt.load_state_dict(state['optimizer'])
    for live, old in zip(opt.param_groups, state['optimizer']['param_groups']):
        for p, index in zip(live['params'], old['params']):
            if index in state['optimizer']['state']:
                opt.state[p] = tree_to(state['optimizer']['state'][index], p.device)
    old_max = max(g['lr'] for g in opt.param_groups)
    for g in opt.param_groups:
        g['lr'] *= lr / old_max
    return opt


def init_run(out, name, config):
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults = json.loads((HERE.parent / 'wandb_config.json').read_text())
    opts = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                entity=os.environ.get('WANDB_ENTITY', defaults['entity']), name=name,
                group='zapformer_lr_fs_diagnostics', dir=str(out),
                mode=os.environ.get('WANDB_MODE', defaults.get('mode', 'online')), config=config)
    try:
        run = init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode'] = 'offline'
        run = init_evaluation(**opts)
    (out / 'wandb_run.json').write_text(json.dumps(dict(id=run.id, url=run.url,
                                                    mode=run.settings.mode)) + '\n')
    return run


def gradient_relation(a, b):
    a = a.double(); b = b.double()
    an = float(a.square().sum().sqrt()); bn = float(b.square().sum().sqrt())
    return dict(ottc_norm=an, forward_sum_norm=bn, fs_to_ottc_norm=bn/max(an, 1e-30),
                cosine=float((a*b).sum()) / max(an*bn, 1e-30))


def task_gradient_probe(state, batch):
    net = new_model(state, 1.)
    net.train(); set_count(net, state['progress']['global_step'])
    captured = {}
    h1 = net.acoustic.ctc_output.register_forward_pre_hook(
        lambda module, args: captured.update(encoder_output=args[0]))
    h2 = net.acoustic.ctc_output.register_forward_hook(
        lambda module, args, result: captured.update(state_logits=result))
    h3 = net.forward_sum.register_forward_hook(
        lambda module, args, result: captured.update(fs=result))
    original = task_adapter.batched_ottc_loss_bucketized
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['ottc'] = result[0]
        return result
    task_adapter.batched_ottc_loss_bucketized = observed
    try:
        reset()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, comp = net(*tensors(batch), aux_loss_scale=0.)
        inputs = (captured['state_logits'], captured['encoder_output'])
        a = torch.autograd.grad(captured['ottc'], inputs, retain_graph=True)
        b = torch.autograd.grad(captured['fs'], inputs)
        result = dict(step=state['progress']['global_step'],
                      components={k: float(v) for k,v in comp.items()},
                      state_logits=gradient_relation(a[0], b[0]),
                      shared_encoder_output=gradient_relation(a[1], b[1]),
                      protocol='Same train-mode BF16 forward, four fixed real Libri100 utterances, no SpecAugment; task gradients stop at output logits/shared encoder representation, excluding encoder auxiliary backward penalties.')
    finally:
        task_adapter.batched_ottc_loss_bucketized = original
        for h in (h1,h2,h3): h.remove()
    del net, captured, a, b
    gc.collect(); torch.cuda.empty_cache()
    return result


def parameter_changes(net, before):
    result = {}
    for kind in ['frontend', 'encoder', 'state_head', 'mass_head']:
        names = [name for name in before if group(name)==kind]
        now = dict(net.named_parameters())
        scale = sum(float(before[n].double().square().sum()) for n in names)
        delta = sum(float((now[n].detach()-before[n]).double().square().sum()) for n in names)
        result[kind] = dict(parameter_l2=math.sqrt(scale), delta_l2=math.sqrt(delta),
                            relative_delta=math.sqrt(delta/max(scale,1e-30)))
    return result


def main():
    if (HERE/'report.json').exists() or (HERE/'cases').exists():
        raise RuntimeError('Refusing to append a fresh diagnostic to existing results; use a new directory.')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; enable()
    ids = json.loads((PRIOR/'gpu_report.json').read_text())['probe_inputs']['train']
    batch = Inputs(jsd_rows()).batch(ids)
    state = torch.load(PRIOR/'origin.pt', map_location='cpu', mmap=True, weights_only=False)
    report = dict(source_sha256=sha(__file__), origin_sha256=sha(PRIOR/'origin.pt'),
                  seed=42, fixed_utterances=ids, steps=STEPS, formal_training_changed=False,
                  protocol='Full native Zapformer, original upstream OTTC/SP/mass unchanged. Same epoch10 model and Rubik state, same RNG/no SpecAugment/four real utterances. Constant LR .01711896 vs twice that; FS weight 0 vs 1. BF16 without AMP loss scaling, unscaled native aux=.05; native module batch-count fixed at origin step1060. Diagnostic comparisons, not the distributed training recipe.',
                  implementation_hashes={str(Path(m.__file__).resolve()): sha(m.__file__)
                                         for m in (native, task_adapter)},
                  task_gradients={}, cases={})
    probe_out=HERE/'gradient_probe'; probe_out.mkdir(parents=True)
    run=init_run(probe_out,'zapformer_lr_fs_task_gradients',dict(diagnostic=True, **{k:v for k,v in report.items() if k not in ('task_gradients','cases')}))
    for label, path in [('origin_epoch10', PRIOR/'origin.pt'), ('jsd_fs_step2250', PRIOR/'jsd_fs.pt')]:
        s=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        result=task_gradient_probe(s,batch); result['checkpoint_sha256']=sha(path)
        report['task_gradients'][label]=result
        run.log({label+'/'+level+'/'+key:value for level in ('state_logits','shared_encoder_output')
                 for key,value in result[level].items()})
        print(json.dumps({'gradient_probe':label,**result}),flush=True)
    run.finish()
    (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    reference=None
    for factor in (1.,2.):
        for fs in (0.,1.):
            label=f'lr{int(factor)}x_fs{int(fs)}'
            out=HERE/'cases'/label; out.mkdir(parents=True)
            lr=BASE_LR*factor
            net=new_model(state,fs); net.train();set_count(net,1060)
            opt=restored_optimizer(net,state,lr)
            run=init_run(out,'zapformer_fixed4_'+label,dict(diagnostic=True,seed=42,origin_sha256=report['origin_sha256'],
                         source_sha256=report['source_sha256'],fixed_utterances=ids,steps=STEPS,lr=lr,
                         optimizer='Native restored BatchedRubik',aux_loss_scale=.05,no_specaugment=True,
                         formal_training_changed=False,loss_config=net.loss_config,protocol=report['protocol']))
            reset()  # W&B initialization must not perturb the shared diagnostic RNG.
            history=[]; changes={}
            for step in range(STEPS+1):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,comp=net(*tensors(batch),aux_loss_scale=.05)
                record=dict(diagnostic_step=step,optimizer_step=1060+step,epoch=10,
                            lr=lr,utterances=4,loss=float(loss.detach()),**{k:float(v) for k,v in comp.items()})
                assert all(math.isfinite(v) for v in record.values())
                history.append(record)
                if step==0:
                    if reference is None: reference=record['ottc_sum']
                    assert record['ottc_sum']==reference, 'Four cases do not start from exactly the same OTTC loss'
                with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                run.log(record)
                if step%20==0: print(json.dumps({'case':label,**record}),flush=True)
                if step==STEPS:break
                before={n:p.detach().clone() for n,p in net.named_parameters()} if step in (0,STEPS-1) else None
                loss.backward();opt.step()
                if before is not None:
                    changes[str(step+1)]=parameter_changes(net,before); del before
            result=dict(lr=lr,fs_weight=fs,first=history[0],last=history[-1],
                        first_last20_ottc_mean=[sum(r['ottc_sum'] for r in part)/len(part)
                                              for part in (history[:20],history[-20:])],
                        parameter_updates=changes,history=history,final_output=output_stats(net,batch),
                        wandb=json.loads((out/'wandb_run.json').read_text()))
            report['cases'][label]=result
            (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            import wandb
            artifact=wandb.Artifact('zapformer-lr-fs-'+label,type='diagnostic')
            artifact.add_file(str(out/'updates.jsonl'));artifact.add_file(str(HERE/'diagnose.py'))
            run.log_artifact(artifact);run.finish()
            del net,opt; gc.collect();torch.cuda.empty_cache()
    (HERE/'COMPLETE.json').write_text(json.dumps(dict(source_sha256=sha(__file__),cases=list(report['cases']),steps=STEPS))+'\n')


if __name__=='__main__': main()
