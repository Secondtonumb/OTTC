"""Paired fixed-data probes of true105, pooled313, and fresh native SP initialization."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'zapformer_loss_diagnosis_20261013'
sys.path.insert(0,str(HERE))
from monophone_model import torch,Monophone,merge_states,native,parameters
sys.path.insert(0,str(PRIOR))
from probe import reset,tensors,sha,enable,Inputs,jsd_rows,group,ConfigurableS3T2,validate_config
from checkpoint import tree_to
import numpy as np
from ifmdd_losses import subphonetic_greedy_complete
import jiwer

BASE_LR=.017118957847577866
CASES=[('mono105_seed42_lr1','phone105','random',BASE_LR),
       ('mono105_seed42_lr2','phone105','random',BASE_LR*2),
       ('pooled313_jsd10_lr1','pooled313','checkpoint10',BASE_LR),
       ('pooled313_jsd10_lr2','pooled313','checkpoint10',BASE_LR*2),
       ('sp313_seed42_lr2','sp313','random',BASE_LR*2)]


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda'])


def tensor_hashes(net):
    result={}
    for kind in ['frontend','encoder','state_head','mass_head']:
        digest=hashlib.sha256()
        for name,p in net.named_parameters():
            if group(name)==kind:
                digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        result[kind]=digest.hexdigest()
    return result


def batch_count(net,step):
    p=parameters();p.update(batch_idx_train=step,max_duration=640,world_size=8)
    native.set_batch_count(net,native.get_adjusted_batch_count(p))


def build(variant,initialization,lr,origin):
    reset();np.random.seed(42)
    if variant=='sp313':
        net=ConfigurableS3T2('jsd',topology='s3_t2_star_star',loss_config=validate_config(dict(ottc_cost='jsd_upstream',forward_sum_weight=0.)))
    else:net=Monophone(variant)
    if initialization=='checkpoint10':net.load_state_dict(origin['model'],strict=True)
    hashes=tensor_hashes(net)
    net.cuda()
    opt=native.BatchedRubik(native.get_parameter_groups_with_lrs(net,lr=lr,include_names=False),lr=lr,beta1=.99)
    if initialization=='checkpoint10':
        opt.load_state_dict(origin['optimizer'])
        for live,old in zip(opt.param_groups,origin['optimizer']['param_groups']):
            for p,index in zip(live['params'],old['params']):
                if index in origin['optimizer']['state']:
                    opt.state[p]=tree_to(origin['optimizer']['state'][index],p.device)
        maximum=max(g['lr'] for g in opt.param_groups)
        for g in opt.param_groups:g['lr']*=lr/maximum
    batch_count(net,1060 if initialization=='checkpoint10' else 0)
    return net,opt,hashes


def collapse(path):
    return [p for p in torch.unique_consecutive(path).tolist() if 3<=p<=102]


def recognition(hyps,refs):
    totals=dict(reference_phones=0,substitutions=0,insertions=0,deletions=0)
    for hyp,ref in zip(hyps,refs):
        ref=[p for p in ref if 3<=p<=102]
        edits=jiwer.process_words(' '.join(map(str,ref)),' '.join(map(str,hyp)))
        totals['reference_phones']+=len(ref)
        for key in ['substitutions','insertions','deletions']:totals[key]+=getattr(edits,key)
    totals['PER_percent']=100*sum(totals[k] for k in ['substitutions','insertions','deletions'])/totals['reference_phones']
    return dict(**totals,predicted_ids=hyps,reference_ids=refs)


def output_stats(net,batch):
    previous=net.training;saved_rng=rng_state();net.eval()
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            features,lengths,tokens,tl=tensors(batch)
            enc,ol=net.acoustic.forward_encoder(features,lengths,aux_loss_scale=0.)
            raw=net.acoustic.ctc_output(enc).float()
            phones=merge_states(raw) if raw.shape[-1]==313 else raw
            phones=phones.index_fill(-1,torch.tensor([1,2],device='cuda'),-1e4)
            valid=torch.arange(phones.shape[1],device='cuda')[None]<lengths[:,None]
            with torch.autocast('cuda',enabled=False):
                mass=net.weight_head(enc.float()).squeeze(-1).masked_fill(~valid,-torch.inf).softmax(-1)
            p=phones.softmax(-1);path=p.argmax(-1)
            hyps=[collapse(path[i,:int(n)]) for i,n in enumerate(lengths)]
            refs=[r['tokens'] for r in batch['rows']]
            result=dict(phone_marginal_recognition=recognition(hyps,refs),
                        blank_argmax_fraction=float((path[valid]==0).float().mean()),
                        phone_posterior_max_mean=float(p.max(-1).values[valid].mean()),
                        phone_entropy_mean=float(-(p*p.clamp_min(1e-30).log()).sum(-1)[valid].mean()),
                        mass=[dict(max_mass=float(mass[i].max()),
                                   effective_frame_fraction=float((-(mass[i]*mass[i].clamp_min(1e-30).log()).sum()).exp()/n))
                              for i,n in enumerate(lengths)])
            if not isinstance(net,Monophone):
                raw=raw.index_fill(-1,torch.arange(1,7,device='cuda'),-1e4)
                native_hyp=subphonetic_greedy_complete(raw.argmax(-1),ol,'s3_t2_star_star')
                result['native_sp_recognition']=recognition([[p for p in h if 3<=p<=102] for h in native_hyp],refs)
            return result
    finally:net.train(previous);restore_rng(saved_rng)


def init_run(out,config):
    from reproduction.wandb_policy import init_evaluation
    import wandb
    defaults=json.loads((HERE.parent/'wandb_config.json').read_text())
    identity=out/'wandb_run.json';saved=json.loads(identity.read_text()) if identity.exists() else None
    opts=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),entity=os.environ.get('WANDB_ENTITY',defaults['entity']),
              group='zapformer_monophone_diagnosis',name=config['case'],dir=str(out),
              mode=os.environ.get('WANDB_MODE',defaults['mode']),config=config)
    if saved:opts.update(id=saved['id'],resume='allow')
    try:run=init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode']='offline';opts.pop('resume',None);run=init_evaluation(**opts)
    identity.write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode))+'\n')
    return run


def invariants():
    reset()
    x=torch.randn(2,7,313,requires_grad=True)
    merged=merge_states(x)
    p=x.softmax(-1)
    expected=torch.cat([p[...,:1],p[...,1:].reshape(2,7,104,3).sum(-1)],-1)
    torch.testing.assert_close(merged.softmax(-1),expected,rtol=1e-5,atol=1e-7)
    merged.sum().backward();assert torch.isfinite(x.grad).all()
    # Check all trainable tensors can be loaded unchanged into the pooled model.
    origin=torch.load(PRIOR/'origin.pt',map_location='cpu',mmap=True,weights_only=False)
    pooled=Monophone('pooled313');pooled.load_state_dict(origin['model'],strict=True)
    assert all(torch.equal(v,origin['model'][n]) for n,v in pooled.state_dict().items())
    reset();mono=Monophone('phone105');reset()
    sp=ConfigurableS3T2('jsd',topology='s3_t2_star_star',loss_config=validate_config(dict(ottc_cost='jsd_upstream',forward_sum_weight=0.)))
    assert mono.acoustic.ctc_output[0].out_features==105
    mh,sh=tensor_hashes(mono),tensor_hashes(sp)
    assert all(mh[k]==sh[k] for k in ['frontend','encoder','mass_head'])
    result=dict(pooled_posterior_matches_state_probability_sum=True,pooled_state_dict_bitwise_origin=True,
                true105_and_sp_seed42_backbone_and_mass_bitwise_equal=True,
                true105_head_shape=list(mono.acoustic.ctc_output[0].weight.shape),
                mono_initial_hashes=mh,sp_initial_hashes=sh,original_ottc_sha256=sha(HERE.parent/'zapformer_s3t2ss_native_20261012/archived_ot_loss.py'))
    (HERE/'INVARIANTS_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    del pooled,mono,sp;gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=120);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;enable()
    invariant=invariants()
    ids=json.loads((PRIOR/'gpu_report.json').read_text())['probe_inputs']['train']
    batch=Inputs(jsd_rows()).batch(ids)
    origin=torch.load(PRIOR/'origin.pt',map_location='cpu',mmap=True,weights_only=False)
    sources={str(p):sha(p) for p in [HERE/'monophone_model.py',HERE/'diagnose.py',Path(native.__file__),
                                     HERE.parent/'zapformer_s3t2ss_native_20261012/archived_ot_loss.py']}
    report=dict(diagnostic=True,formal_training_changed=False,steps=args.steps,seed=42,fixed_utterances=ids,
                original_ottc_formula_unchanged=True,source_hashes=sources,invariants=invariant,cases={},
                protocol='Native full Zapformer/BatchedRubik/BF16; no SpecAugment, no FS, unscaled aux=.05; fixed batch-count at respective initialization. True105 vs SP have bitwise identical seed42 backbone/mass. Pooled313 uses identical epoch10 model/optimizer. Raw loss across inventories is not directly comparable; recognition uses same fixed native-phone sequences.')
    for label,variant,initialization,lr in CASES:
        out=HERE/'cases'/label;out.mkdir(parents=True,exist_ok=True)
        complete=out/'COMPLETE.json'
        if complete.exists():
            report['cases'][label]=json.loads(complete.read_text());continue
        config=dict(case=label,variant=variant,initialization=initialization,lr=lr,seed=42,steps=args.steps,
                    checkpoint_sha256=sha(PRIOR/'origin.pt') if initialization=='checkpoint10' else None,
                    loss='archived original OTTC-JSD, single-phone targets' if variant!='sp313' else 'archived original SPOTTC-JSD',
                    target_states_per_phone=3 if variant=='sp313' else 1,head_dimension=105 if variant=='phone105' else 313,
                    forward_sum_weight=0.,aux_loss_scale=.05,no_specaugment=True,
                    diagnostic=True,formal_training_changed=False,fixed_utterances=ids,source_hashes=sources)
        net,opt,hashes=build(variant,initialization,lr,origin)
        net.train();start=0;history=[];initial_output=None;identity=None
        checkpoint=out/'checkpoint.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['config']==config,'Refusing resume after a config/source change'
            net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            for live,old in zip(opt.param_groups,saved['optimizer']['param_groups']):
                for p,index in zip(live['params'],old['params']):
                    if index in saved['optimizer']['state']:opt.state[p]=tree_to(saved['optimizer']['state'][index],p.device)
            start=saved['step'];history=saved['history'];initial_output=saved['initial_output'];resume_rng=saved['rng']
            # Preserve uncommitted logs separately before replaying from the
            # durable checkpoint. The committed history has no duplicate steps.
            recorded=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
            tail=[r for r in recorded if r['diagnostic_step']>=start]
            if tail:
                replay=out/f'uncommitted_before_resume_step{start}.jsonl'
                if replay.exists():raise RuntimeError('An earlier replay exists; inspect it before resuming again')
                replay.write_text(''.join(json.dumps(r)+'\n' for r in tail))
            (out/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        elif (out/'updates.jsonl').exists():raise RuntimeError('History exists without a checkpoint; use a new output directory')
        run=init_run(out,config)
        if start:restore_rng(resume_rng)
        else:reset();np.random.seed(42)
        if initial_output is None:initial_output=output_stats(net,batch)
        for step in range(start,args.steps+1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,comp=net(*tensors(batch),aux_loss_scale=.05)
            record=dict(diagnostic_step=step,optimizer_step=step,epoch=0,lr=lr,utterances=len(ids),
                        loss=float(loss.detach()),loss_per_utterance=float(loss.detach()/len(ids)),
                        **{k:float(v) for k,v in comp.items()})
            assert math.isfinite(record['loss'])
            if not history or history[-1]['diagnostic_step']!=step:
                history.append(record)
                with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                run.log(record)
            if step%20==0:print(json.dumps({'case':label,**record}),flush=True)
            if step==args.steps:break
            loss.backward();opt.step()
            if (step+1)%40==0:
                payload=dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),step=step+1,
                             config=config,history=history,initial_output=initial_output)
                temporary=out/'checkpoint.partial.pt';torch.save(payload,temporary);temporary.replace(checkpoint)
        final=output_stats(net,batch)
        result=dict(config=config,initial_parameter_hashes=hashes,first=history[0],last=history[-1],history=history,
                    initial_output=initial_output,final_output=final,wandb=json.loads((out/'wandb_run.json').read_text()))
        complete.write_text(json.dumps(result,indent=2)+'\n')
        report['cases'][label]=result
        (HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        run.log({'fixed_train/phone_marginal_PER_percent':final['phone_marginal_recognition']['PER_percent'],
                 'fixed_train/blank_argmax_fraction':final['blank_argmax_fraction']})
        import wandb
        artifact=wandb.Artifact('zapformer-monophone-'+label,type='diagnostic')
        for path in [complete,out/'updates.jsonl',HERE/'monophone_model.py',HERE/'diagnose.py',HERE/'INVARIANTS_PASS.json']:
            artifact.add_file(str(path))
        run.log_artifact(artifact);run.finish()
        print(json.dumps({'case_complete':label,'initial_loss':history[0]['loss'],'final_loss':history[-1]['loss'],
                          'initial_PER':initial_output['phone_marginal_recognition']['PER_percent'],
                          'final_PER':final['phone_marginal_recognition']['PER_percent']}),flush=True)
        del net,opt;gc.collect();torch.cuda.empty_cache()
    (HERE/'COMPLETE.json').write_text(json.dumps(dict(cases=list(report['cases']),steps=args.steps,source_hashes=sources))+'\n')


if __name__=='__main__':main()
