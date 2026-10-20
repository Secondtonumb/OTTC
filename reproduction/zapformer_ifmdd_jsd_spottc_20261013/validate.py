"""Native-loop integration, resume replay, and a bounded real-speech pilot."""
from pathlib import Path
import copy
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from task_ottc import TopologyOTTC, decode, native, NATIVE
from fsa_decode import graph, decode_fsa, INVENTORY
from data import Inputs, jsd_rows
from k2_determinism import enable


def run(directory, steps, tiny=True):
    checkpoints=sorted(directory.glob('step-*.pt'))
    if checkpoints and int(checkpoints[-1].stem.split('-')[1])>=steps:
        return
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
               str(HERE/'train_ottc.py'), '--output', str(directory), '--epochs', '3',
               '--smoke', '--stop-after', str(steps), '--max-duration', '100',
               '--save-every', '1', '--eval-every', '1000', '--average-period', '2',
               '--keep-last', '0', '--eval-limit', '4']
    if tiny:
        command.append('--tiny')
    subprocess.run(command, check=True)


def compare(left, right):
    a = torch.load(left, map_location='cpu', weights_only=False)
    b = torch.load(right, map_location='cpu', weights_only=False)
    maximum = {}
    def visit(x, y, path):
        if torch.is_tensor(x):
            assert x.dtype == y.dtype and x.shape == y.shape, path
            if x.is_floating_point():
                delta = float((x-y).abs().max()) if x.numel() else 0.
                maximum[path.split('/')[0]] = max(delta, maximum.get(path.split('/')[0], 0.))
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6, msg=path)
            else:
                assert torch.equal(x, y), path
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for key in x:
                visit(x[key], y[key], path+'/'+str(key))
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y)):
                visit(u, v, path+'/'+str(i))
        else:
            assert x == y, (path, x, y)
    for key in ('model', 'model_avg', 'optimizer', 'scheduler', 'contract'):
        visit(a[key], b[key], key)
    for key in ('global_step', 'epoch', 'step_in_epoch', 'optimizer_updates'):
        visit(a['progress'][key], b['progress'][key], 'progress/'+key)
    for i, (x, y) in enumerate(zip(a['rank_states'], b['rank_states'])):
        visit(x['sampler'], y['sampler'], f'sampler/{i}')
        visit(x['scaler'], y['scaler'], f'scaler/{i}')
        for key in ('cpu', 'cuda', 'specaug_seed', 'specaug_generators'):
            visit(x['random'][key], y['random'][key], f'rng/{i}/{key}')
    return maximum


def edits(ref, hyp):
    import jiwer
    e = jiwer.process_words(' '.join(map(str, ref)), ' '.join(map(str, hyp)))
    return dict(reference_phones=len(ref), substitutions=e.substitutions,
                deletions=e.deletions, insertions=e.insertions,
                PER_percent=100 * (e.substitutions+e.deletions+e.insertions)/len(ref))


def overfit(base, steps):
    """Fixed train utterances: diagnose learnability, never a held-out PER."""
    from reproduction.wandb_policy import init_evaluation
    out=base/'fixed4'
    if (out/'results.json').exists():
        result=json.loads((out/'results.json').read_text())
        if result['config']['steps']!=steps:
            raise ValueError('Choose a new TEST_OUTPUT when changing OVERFIT_STEPS')
        return result
    if (out/'updates.jsonl').exists():
        raise ValueError('Interrupted fixed4 pilot: choose a new TEST_OUTPUT to avoid overwriting its history')
    torch.manual_seed(42); random.seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    enable()
    rows = sorted(jsd_rows(), key=lambda r: (r['frames'], r['id']))
    # Real short utterances with sufficient speech, retained unmodified MFA targets.
    selected = [r for r in rows if 180 <= r['frames'] <= 450 and len(r['tokens']) >= 12][:4]
    assert len(selected) == 4
    batch = Inputs(selected).batch([r['id'] for r in selected])
    args = (batch['inputs'].cuda(), batch['supervisions']['num_frames'].cuda(),
            batch['tokens'].cuda(), batch['token_lengths'].cuda())
    net = TopologyOTTC().cuda()
    groups = native.get_parameter_groups_with_lrs(net, lr=.02, include_names=False)
    optimizer = native.BatchedRubik(groups, lr=.02, beta1=.99)
    out = base/'fixed4'; out.mkdir(parents=True, exist_ok=True)
    import wandb
    defaults = json.loads((HERE.parent/'wandb_config.json').read_text())
    identity = out/'wandb_run.json'
    saved = json.loads(identity.read_text()) if identity.exists() else {}
    config = dict(diagnostic=True, held_out=False, fixed_training_utterances=[r['id'] for r in selected],
                  objective='explicit IF-MDD standard-JSD subphonetic transport sum', inventory='native MFA IPA105 targets, sorted IF-MDD303 substate outputs including blank; silence one state',
                  model='full native Zapformer topology303 + learned mass head', seed=42,
                  optimizer='native BatchedRubik .02 beta1 .99, native relative parameter learning rates',
                  steps=steps, augmentation=False, encoder_aux_loss_scale=0.,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    opts = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
                mode=os.environ.get('WANDB_MODE', defaults.get('mode', 'online')),
                name='zapformer_ifmdd_jsd_spottc_fixed4', group='zapformer_ifmdd_jsd_spottc_20261013',
                id=saved.get('id'), resume='allow' if saved else None, dir=str(out), config=config,
                settings=wandb.Settings(init_timeout=45))
    try:
        run = init_evaluation(**opts)
    except wandb.errors.CommError:
        opts['mode']='offline'; opts.pop('resume',None); run=init_evaluation(**opts)
    identity.write_text(json.dumps(dict(id=run.id,url=run.url,mode=run.settings.mode))+'\n')
    run.define_metric('optimizer_step')
    run.define_metric('train/*', step_metric='optimizer_step')
    history=[]; snapshots=[]
    for step in range(steps+1):
        net.train(); optimizer.zero_grad(set_to_none=True)
        native.set_batch_count(net, step)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, comp = net(*args, aux_loss_scale=0.)
        assert math.isfinite(float(loss))
        record=dict(optimizer_step=step, epoch=step, lr=max(g['lr'] for g in optimizer.param_groups),
                    raw_loss=float(loss.detach()), **{k:float(v) for k,v in comp.items()})
        history.append(record)
        with (out/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        run.log(dict(optimizer_step=step, **{'train/'+k:v for k,v in record.items()}))
        if step % 50 == 0 or step == steps:
            net.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores, lengths, mass_logits = net.acoustic_logits(args[0],args[1])
                hypotheses = decode(scores,lengths,net.loss_module)
                vector, text = graph(net.loss_module, device=scores.device)
                fsa_decodes=[decode_fsa(scores[i,:int(length)],vector) for i,length in enumerate(lengths)]
            stats=[]
            for row,hyp,fsa in zip(selected,hypotheses,fsa_decodes):
                fhyp,state_path,starts=fsa
                stats.append(dict(id=row['id'],reference=row['tokens'],hypothesis=hyp,
                                  reference_labels=[next(label for label,idx in INVENTORY.items() if idx==p) for p in row['tokens']],
                                  hypothesis_labels=[next(label for label,idx in INVENTORY.items() if idx==p) for p in hyp],
                                  fsa_hypothesis=fhyp,fsa_edits=edits(row['tokens'],fhyp),
                                  fsa_state_path=state_path,
                                  fsa_state_labels=[net.loss_module.labels[p] for p in state_path],
                                  **edits(row['tokens'],hyp)))
            snapshots.append(dict(optimizer_step=step,utterances=stats))
            n=sum(x['reference_phones'] for x in stats)
            per=100*sum(x[k] for x in stats for k in ('insertions','deletions','substitutions'))/n
            fsa_per=100*sum(x['fsa_edits'][k] for x in stats for k in ('insertions','deletions','substitutions'))/n
            run.log({'optimizer_step':step,'train/fixed4_PER_percent':per,'train/fixed4_FSA_PER_percent':fsa_per})
            print(json.dumps(dict(fixed4_step=step,loss=record['raw_loss'],PER_percent=per,FSA_PER_percent=fsa_per)),flush=True)
        if step == steps:
            break
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
    final = dict(config=config,first=history[0],last=history[-1],snapshots=snapshots,
                 interpretation='Fixed train utterance PER is a learnability diagnostic, not a benchmark or generalization result.')
    (out/'reference_decode.json').write_text(json.dumps(snapshots,indent=2,ensure_ascii=False)+'\n')
    (out/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    torch.save(dict(model=net.cpu().state_dict(),optimizer=optimizer.state_dict(),
                    progress=dict(optimizer_updates=steps),contract=config),out/f'step-{steps:09d}.pt')
    artifact=wandb.Artifact('zapformer-ifmdd-topology-ottc-fixed4',type='diagnostic')
    for path in (out/'results.json',out/'updates.jsonl',HERE/'reference.json',HERE/'task_ottc.py'):
        artifact.add_file(str(path))
    run.log_artifact(artifact);run.summary['diagnostic/complete']=True;run.finish()
    return final


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',type=Path,default=HERE/'validation')
    ap.add_argument('--overfit-steps',type=int,default=200)
    ap.add_argument('--pilot-only',action='store_true')
    args=ap.parse_args()
    if args.overfit_steps<1:ap.error('overfit steps must be positive')
    base=args.output_root.resolve();base.mkdir(parents=True,exist_ok=True)
    if (base/'INTEGRATION_PASS.json').exists():
        print('Validation already complete: '+str(base/'INTEGRATION_PASS.json'));return
    subprocess.run([sys.executable,str(HERE/'check_parity.py'),'--output',str(base/'PARITY_PASS.json'),'--device','cuda'],check=True)
    maximum={};a=[];b=[]
    if not args.pilot_only:
        run(base/'uninterrupted',2)
        run(base/'resumed',1)
        run(base/'resumed',2)
        maximum=compare(base/'uninterrupted/step-000000002.pt',base/'resumed/step-000000002.pt')
        a=[json.loads(line)['loss'] for line in (base/'uninterrupted/updates.jsonl').open()]
        b=[json.loads(line)['loss'] for line in (base/'resumed/updates.jsonl').open()]
        assert a==b,(a,b)
        run(base/'full',1,tiny=False)
    torch.cuda.empty_cache()
    pilot=overfit(base,args.overfit_steps)
    result=dict(status='passed', native_loop='not run' if args.pilot_only else 'tiny and full optimizer updates completed',
                resume='not run' if args.pilot_only else 'optimizer/model/average/scheduler and RNG replay matched',
                max_absolute_difference=maximum, uninterrupted_losses=a, resumed_losses=b,
                fixed_train_loss_first=pilot['first']['raw_loss'],fixed_train_loss_last=pilot['last']['raw_loss'],
                held_out_benchmark_completed=False,objective='ifmdd_standard_jsd_spottc',overfit_steps=args.overfit_steps)
    ce=json.loads((HERE.parent/'zapformer_ifmdd_topology_ottc_20261013/validation/fixed4/results.json').read_text())
    result['matching_ce_reference']={'reused':True,'source':str(HERE.parent/'zapformer_ifmdd_topology_ottc_20261013/validation/fixed4/results.json'),'scope':'same seed42 native Zapformer, data and optimizer; different cost raw losses are not directly comparable','final_snapshot':ce['snapshots'][-1]}
    (base/'INTEGRATION_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
