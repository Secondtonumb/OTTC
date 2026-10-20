"""Isolate auxiliary-gradient scale on the same four real utterances and warm start."""
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from probe import (torch,native,parameters,ConfigurableS3T2,validate_config,Inputs,jsd_rows,
                   reset,tensors,sha,enable,output_stats)
from checkpoint import tree_to


def main():
    from reproduction.wandb_policy import init_evaluation
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;enable()
    import wandb
    defaults=json.loads((HERE.parent/'wandb_config.json').read_text())
    ids=json.loads((HERE/'gpu_report.json').read_text())['probe_inputs']['train']
    rows=jsd_rows();batch=Inputs(rows).batch(ids)
    mean_frames=float(batch['supervisions']['num_frames'].float().mean())
    state=torch.load(HERE/'origin.pt',map_location='cpu',mmap=True,weights_only=False)
    config=validate_config(dict(ottc_cost='jsd_upstream',forward_sum_weight=1.))
    outcomes={}
    for condition,aux in [('native_aux',.05),('frame_normalized_aux',.05/mean_frames)]:
        reset();net=ConfigurableS3T2('jsd',topology='s3_t2_star_star',loss_config=config).cuda()
        net.load_state_dict(state['model'],strict=True)
        groups=native.get_parameter_groups_with_lrs(net,lr=.02,include_names=False)
        opt=native.BatchedRubik(groups,lr=.02,beta1=.99)
        opt.load_state_dict(state['optimizer'])
        for live,old in zip(opt.param_groups,state['optimizer']['param_groups']):
            for parameter,index in zip(live['params'],old['params']):
                if index in state['optimizer']['state']:
                    opt.state[parameter]=tree_to(state['optimizer']['state'][index],parameter.device)
        # Keep the inherited relative native learning-rate groups, then use the
        # same explicit maximum .01711896 for both isolated diagnostics.
        old_max=max(g['lr'] for g in opt.param_groups)
        for g in opt.param_groups:g['lr']*=.017118957847577866/old_max
        params=parameters();params.update(batch_idx_train=1060,max_duration=640,world_size=8)
        native.set_batch_count(net,native.get_adjusted_batch_count(params))
        net.train();reset()
        out=HERE/'overfit'/condition;out.mkdir(parents=True,exist_ok=True)
        identity=out/'wandb_run.json';saved=json.loads(identity.read_text()) if identity.exists() else {}
        opts=dict(project=defaults['project'],entity=defaults['entity'],name='zapformer_fixed4_'+condition,
                  group='zapformer_diagnostics',dir=str(out),id=saved.get('id'),resume='allow' if saved else None,
                  mode=os.environ.get('WANDB_MODE',defaults.get('mode','online')),
                  config=dict(diagnostic=True,formal_training_changed=False,seed=42,origin_sha256=sha(HERE/'origin.pt'),
                              source_sha256=sha(__file__),fixed_utterances=ids,no_specaugment=True,
                              aux_loss_scale=aux,aux_frame_mean=mean_frames,loss_config=config,
                              optimizer='Native Rubik restored from epoch10; identical group LR max .017118957847577866',
                              steps=120))
        try:run=init_evaluation(**opts)
        except wandb.errors.CommError:
            opts['mode']='offline';opts.pop('resume',None);run=init_evaluation(**opts)
        identity.write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode))+'\n')
        history=[]
        for step in range(121):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,comp=net(*tensors(batch),aux_loss_scale=aux)
            record=dict(diagnostic_step=step,loss=float(loss.detach()),loss_per_utterance=float(loss.detach()/4),
                        **{k:float(v) for k,v in comp.items()})
            assert math.isfinite(record['loss'])
            history.append(record)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            run.log(record)
            if step%20==0:print(json.dumps({'condition':condition,**record}),flush=True)
            if step==120:break
            loss.backward();opt.step()
        stats=output_stats(net,batch)
        outcomes[condition]=dict(aux_loss_scale=aux,first=history[0],last=history[-1],history=history,final_output=stats)
        (HERE/'overfit_report.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        artifact=wandb.Artifact('zapformer-fixed4-'+condition,type='diagnostic')
        artifact.add_file(str(out/'updates.jsonl'));artifact.add_file(str(HERE/'overfit.py'))
        run.log_artifact(artifact);run.finish()
        del net,opt;gc.collect();torch.cuda.empty_cache()
    (HERE/'OVERFIT_COMPLETE.json').write_text(json.dumps(dict(source_sha256=sha(__file__),conditions=list(outcomes),mean_frames=mean_frames,steps=120))+'\n')


if __name__=='__main__':main()
