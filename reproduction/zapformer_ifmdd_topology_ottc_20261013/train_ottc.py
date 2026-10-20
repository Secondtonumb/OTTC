"""Execute native train_one_epoch; adapt only phone loss, inputs, tracking and resume."""
import sys
from pathlib import Path
NATIVE = Path(__file__).resolve().parent.parent / "zapformer_s3t2ss_native_20261012"
sys.path.insert(0, str(NATIVE))
import runtime
import argparse
import builtins
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import train as native
from task_model import parameters
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_ottc import TopologyOTTC
from augmentation import augment
from data import CE_ROOT, JSD_MANIFEST, Inputs, Loader, plan, read_rows, jsd_rows
import checkpoint as persistence
from k2_determinism import enable as enable_deterministic_k2

HERE = Path(__file__).resolve().parent


class PhoneTokenizer:
    """Native failure diagnostics expect an encode() API, even for phones."""
    def encode(self, texts, out_type=int):
        return [[out_type(token) for token in text.split()] for text in texts]


class TrainableView:
    """Give native ParamPlotter the parameters with gradients in this task."""
    def __init__(self, model):
        self.model = model.module if isinstance(model, DDP) else model

    def parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tracking:
    def __init__(self, output, contract, rank, smoke=False):
        self.output, self.rank, self.run = output, rank, None
        if rank:
            return
        import wandb
        defaults = json.loads((HERE.parent/'wandb_config.json').read_text())
        identity = output/'wandb_run.json'
        saved = json.loads(identity.read_text()) if identity.exists() else {}
        opts = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
                    entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
                    mode='offline' if smoke else os.environ.get('WANDB_MODE', defaults.get('mode','online')),
                    name=output.name, group='zapformer_ifmdd_topology_ottc_20261013', id=saved.get('id'),
                    resume='allow' if saved else None, dir=str(output), config=contract,
                    settings=wandb.Settings(init_timeout=45, disable_code=True))
        try:
            self.run = wandb.init(**opts)
        except wandb.errors.CommError:
            opts['mode']='offline'; opts.pop('resume',None)
            self.run=wandb.init(**opts)
        identity.write_text(json.dumps(dict(id=self.run.id,url=self.run.url,mode=self.run.settings.mode))+'\n')
        self.run.define_metric('optimizer_step')
        for prefix in ('train/*','evaluation/*','native/*'):
            self.run.define_metric(prefix,step_metric='optimizer_step')

    def log(self, record, prefix='train'):
        if self.rank:
            return
        with (self.output/'updates.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        payload={'optimizer_step':record.get('optimizer_step',0)}
        payload.update({prefix+'/'+key:value for key,value in record.items() if isinstance(value,(int,float))})
        self.run.log(payload)
        print(json.dumps(record),flush=True)

    def add_scalar(self, tag, value, global_step):
        if self.rank==0:
            self.run.log({'optimizer_step':global_step,'native/'+tag:float(value)})

    def finish(self, completed=False):
        from reproduction.wandb_policy import log_evaluation_artifact
        if self.rank:
            return
        import wandb
        artifact=wandb.Artifact(self.output.name+'-metrics',type='evaluation')
        for name in ('run_config.json','timit_metrics.jsonl','evaluation_loss.jsonl','progress.json','updates.jsonl'):
            p=self.output/name
            if p.exists():artifact.add_file(str(p))
        log_evaluation_artifact(self.run, artifact)
        self.run.summary['training/complete']=completed
        self.run.finish()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--objective',choices=['ifmdd_ottc_ce'],default='ifmdd_ottc_ce')
    ap.add_argument('--topology',choices=['s3_t2_star_star'],default='s3_t2_star_star')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--init-from',type=Path)
    ap.add_argument('--manifest','--jsd-manifest',dest='manifest',type=Path,default=JSD_MANIFEST)
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--base-lr',type=float,default=.02)
    ap.add_argument('--encoder-aux-loss-scale',type=float,default=0.)
    ap.add_argument('--max-duration',type=float,default=100.)
    ap.add_argument('--save-every',type=int,default=50)
    ap.add_argument('--eval-every',type=int,default=50)
    ap.add_argument('--average-period',type=int,default=200)
    ap.add_argument('--keep-last',type=int,default=8)
    ap.add_argument('--max-wall-seconds',type=float,default=13800.)
    ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--tiny',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--eval-limit',type=int,default=0)
    a=ap.parse_args()
    vendor_manifest=json.loads((NATIVE/'vendor_manifest.json').read_text())
    for name,digest in vendor_manifest['sha256'].items():
        if sha(NATIVE/'vendor'/name)!=digest:
            raise RuntimeError('Frozen native source modified: '+name)
    if min(a.epochs,a.average_period,a.save_every,a.eval_every)<=0:
        ap.error('epochs and intervals must be positive')
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    # Preserve the archived deterministic dependency adapter and native encoder.
    enable_deterministic_k2()
    # Nested auxiliary backward calls can change BF16 accumulation order.
    torch.autograd.set_multithreading_enabled(False)
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    manifest=a.manifest.resolve()
    smoke_rows=max(1024,world*256)
    rows=jsd_rows(smoke_rows if a.smoke else 0,manifest)
    if a.smoke:
        rows=rows[:smoke_rows]
    net=TopologyOTTC(tiny=a.tiny)
    if a.init_from:
        initial=torch.load(a.init_from,map_location='cpu',mmap=True,weights_only=False)
        source=initial['contract']
        if source.get('topology')!=a.topology or source.get('objective')!='ifmdd_ottc_ce':
            raise ValueError('Initial checkpoint must match MFA inventory, encoder and IF-MDD topology OTTC recipe')
        net.load_state_dict(initial['model'],strict=True)
        del initial
    average=copy.deepcopy(net).double() if rank==0 else None
    net.cuda()
    model=DDP(net,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True) if world>1 else net
    groups=native.get_parameter_groups_with_lrs(model,lr=a.base_lr,include_names=False)
    groups=[dict(group,params=[p for p in group['params'] if p.requires_grad]) for group in groups]
    optimizer=native.BatchedRubik([group for group in groups if group['params']],lr=a.base_lr,beta1=.99)
    scaler=native.GradScaler(enabled=True,init_scale=1.)
    initial_plan=plan(rows,a.max_duration,rank,world,0)
    if not initial_plan:raise ValueError('Native sampler produced no complete batches; increase the input fixture')
    counts=[len(initial_plan)]*a.epochs
    all_counts=[None]*world;dist.all_gather_object(all_counts,counts)
    if any(value!=counts for value in all_counts):
        raise ValueError('DDP ranks have unequal epoch lengths')
    scheduler=native.InterpCosineLRScheduler(optimizer,batches_per_epoch=counts,min_factor=.025,linear_scale=.5)
    params=parameters(a.tiny)
    params.update(exp_dir=a.output,cur_epoch=1,world_size=world,max_duration=a.max_duration,
                  use_autocast=True,dtype=torch.bfloat16,aux_loss_scale=a.encoder_aux_loss_scale,
                  average_period=a.average_period,save_every_n=max(a.save_every,a.average_period),keep_last_k=0,
                  log_interval=50,valid_interval=a.eval_every,print_diagnostics=False,inf_check=False,subsampling_factor=1)
    import k2, _k2
    dependency_files=[Path(k2.__file__).parent/name for name in ('dense_fsa_vec.py','autograd.py','ops.py')]+[Path(_k2.__file__)]
    contract=dict(recipe='ifmdd_topology_ottc_on_native_zapformer_v1',topology=a.topology,ottc_cost='CE transport; IF-MDD jsd=False',
                  objective=a.objective,initialization=str(a.init_from) if a.init_from else 'seed42 random',
                  init_sha256=sha(a.init_from) if a.init_from else None,manifest=str(manifest),manifest_sha256=sha(manifest),
                  determinism=dict(policy='user-approved k2 deterministic CUDA index_add and explicit terminal rows',
                      torch_algorithms=True,cudnn=True,autograd_multithreading=False,
                      cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                      torch_version=torch.__version__,cuda_version=torch.version.cuda,k2_version=version('k2'),
                      dependency_sha256={str(p):sha(p) for p in dependency_files}),
                  world_size=world,epochs=a.epochs,base_lr=a.base_lr,max_duration=a.max_duration,
                  optimizer='native BatchedRubik beta1=.99; native parameter grouping',scheduler='native InterpCosineLRScheduler min_factor=.025 linear_scale=.5',
                  epoch_batches=counts,precision='bf16+native GradScaler init_scale1',aux_loss_scale=a.encoder_aux_loss_scale,
                  auxiliary_after_step2000=.25,augmentation='native AlternatingSpecAugment defaults; unchanged sequence targets',
                  phone_objective_scale=1.,copies_per_epoch=1,scaler_resume='materialize restored CUDA scale before first forward',
                  average_period=a.average_period,save_every=a.save_every,eval_every=a.eval_every,
                  keep_last=a.keep_last,
                  native_train_source_sha256=sha(native.__file__),vendor_manifest_sha256=sha(NATIVE/'vendor_manifest.json'),
                  adapter_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
                  scorer_sha256=sha(CE_ROOT.parent/'english_suite/mfa_metrics.py'),
                  timeline_provenance_sha256=None,
                  native_loop_unmodified=True,microbatch=False,gradient_clipping=None,loss_reduction='IF-MDD OTTC sum of per-sentence mass-normalized transport CE; native DDP gradient average',
                  full_asr_reproduction=False,adaptations=['80-bin Fbank / 10ms frame-preserving frontend','MFA labels retained; sorted IF-MDD state inventory; silence one state','unchanged IF-MDD topology OTTC with learned mass floor and logit clamp; transport CE cost','topology branch only; no CTC/content/fused/forward-sum terms','native trainable encoder replaces frozen WavLM','native optimizer, schedule, augmentation, checkpoint averaging and resume'],
                  timit_gradient_training=False,smoke=a.smoke,tiny=a.tiny)
    contract['ifmdd_reference'] = json.loads((HERE/'reference.json').read_text())
    contract['native_adapter_sha256'] = {p.name: sha(p) for p in NATIVE.glob('*.py')}
    checkpoints=sorted(a.output.glob('step-*.pt'))
    saved=torch.load(checkpoints[-1],map_location='cpu',mmap=True,weights_only=False) if checkpoints else None
    epoch=saved['rank_states'][rank]['sampler']['epoch'] if saved else 0
    del saved
    inputs=Inputs(rows)
    loader=Loader(inputs,initial_plan if epoch==0 else plan(rows,a.max_duration,rank,world,epoch),epoch)
    specaug=native.AlternatingSpecAugment(seed=42+50*(epoch+1)+512*rank)
    progress=dict(global_step=0,epoch=0,step_in_epoch=0,optimizer_updates=0,save_every_steps=a.save_every)
    if checkpoints:
        progress=persistence.load(checkpoints[-1],net,average,optimizer,scheduler,scaler,specaug,loader,contract)
        params.batch_idx_train=progress['global_step']
        # Native training statistics influence reporting and are also restored.
        for key,value in progress.get('native_statistics',{}).items():params[key]=value
    else:
        scheduler.set_epoch(1);scheduler.set_batch(0)
        torch.cuda.manual_seed(42+50+512*rank)
    rng=persistence.random_state(specaug)
    tracking=Tracking(a.output,contract,rank,a.smoke)
    persistence.restore_random(rng,specaug)
    if rank==0:
        (a.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    updates=[progress['optimizer_updates']]
    optimizer.register_step_post_hook(lambda *args:updates.__setitem__(0,updates[0]+1))
    state={'last':None,'save_due':False,'saved_step':-1,'started':time.monotonic()}
    # Keep native function bytecode and its update order unchanged. Resume only
    # changes enumerate's start so batch-index schedules keep their real offset.
    native.enumerate=lambda iterable,start=0:builtins.enumerate(iterable,start=getattr(iterable,'batch_offset',start))

    def should_stop():
        flag=torch.tensor(int((a.stop_after and params.batch_idx_train>=a.stop_after) or
                              (a.max_wall_seconds and time.monotonic()-state['started']>=a.max_wall_seconds)),device='cuda')
        dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        return bool(flag)

    def save():
        progress.update(global_step=params.batch_idx_train,step_in_epoch=loader.cursor,optimizer_updates=updates[0],
                        native_statistics={key:params[key] for key in ('best_train_epoch','best_train_loss','best_valid_epoch','best_valid_loss')})
        persistence.save(a.output/f'step-{params.batch_idx_train:09d}.pt',net,average,optimizer,scheduler,scaler,specaug,loader.state_dict(),progress,contract)
        state['saved_step']=params.batch_idx_train;state['save_due']=False
        if rank==0:(a.output/'progress.json').write_text(json.dumps(progress)+'\n')
        if rank==0 and a.keep_last>0:
            for obsolete in sorted(a.output.glob('step-*.pt'))[:-a.keep_last]:obsolete.unlink()

    def boundary():
        if state['last'] is not None:
            record=state.pop('last');state['last']=None
            keys=sorted(record)
            values=torch.tensor([record[key] for key in keys],device='cuda',dtype=torch.float64)
            dist.all_reduce(values)
            record={key:float(value) for key,value in zip(keys,values)}
            record.update(optimizer_step=params.batch_idx_train,epoch=params.cur_epoch,
                          optimizer_updates=updates[0],lr=max(scheduler.get_last_lr()),grad_scale=scaler.get_scale(),
                          loss_per_frame=record['loss']/record['frames'],loss_reduction='sum_all_ranks')
            actual_updates=torch.tensor([updates[0],updates[0]],device='cuda',dtype=torch.long)
            dist.all_reduce(actual_updates[:1],op=dist.ReduceOp.MIN)
            dist.all_reduce(actual_updates[1:],op=dist.ReduceOp.MAX)
            if actual_updates[0]!=actual_updates[1]:
                raise RuntimeError('AMP optimizer skips diverged across DDP ranks')
            tracking.log(record)
        if params.batch_idx_train and (state['save_due'] or params.batch_idx_train%a.save_every==0) and state['saved_step']!=params.batch_idx_train:
            save()

    def compute_loss(params,model,sp,batch,is_training,aux_loss_scale=0.,specaug=None):
        features=batch['inputs'].cuda();lengths=batch['supervisions']['num_frames'].cuda()
        gold=None
        if specaug is not None:
            with torch.autocast('cuda',enabled=False):features,gold=augment(specaug,features.float(),lengths,gold)
        with torch.set_grad_enabled(is_training):
            loss,components=model(features,lengths,batch['tokens'].cuda(),batch['token_lengths'].cuda(),gold,aux_loss_scale=aux_loss_scale)
        info=native.MetricsTracker()
        info['frames']=int(lengths.sum());info['loss']=float(loss.detach())
        info['utterances']=features.shape[0];info['padded_frames']=features.shape[0]*features.shape[1]
        info['valid_ce_frames']=int((gold!=-100).sum()) if gold is not None else 0
        for key,value in components.items():info[key]=float(value)
        if is_training:
            state['last']=dict(info);state['last']['aux_loss_scale']=aux_loss_scale/world
        return loss,info

    metrics_spec=importlib.util.spec_from_file_location('ifmdd_topology_ottc_metrics',HERE/'metrics.py')
    metrics=importlib.util.module_from_spec(metrics_spec);metrics_spec.loader.exec_module(metrics)
    validation_rows=read_rows(CE_ROOT/'eval_data/development.jsonl')
    if a.eval_limit:validation_rows=validation_rows[:a.eval_limit]
    for row in validation_rows:
        row['intervals']=[[p['mfa_id'],p['start'],p['end']] for p in row['phone_intervals']]
    valid_inputs=Inputs(validation_rows)

    def benchmark(split):
        rng=persistence.random_state(specaug);training=net.training
        net.eval()
        try:
            result=metrics.evaluate(net,split,rank,world,a.output/'evaluations'/f'step-{params.batch_idx_train:09d}'/split/f'rank-{rank}',limit=a.eval_limit)
        finally:
            net.train(training);persistence.restore_random(rng,specaug)
        if rank==0:
            record=dict(optimizer_step=params.batch_idx_train,epoch=params.cur_epoch,**result)
            with (a.output/'timit_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            for mode in ('text_free','fsa_text_free','forced_k2','mapped_arpabet','fsa_mapped_arpabet'):
                tracking.run.log(dict(optimizer_step=params.batch_idx_train,**{'evaluation/'+split+'/'+mode+'/'+key:value for key,value in result[mode].items() if isinstance(value,(int,float))}))

    def averaged_benchmark():
        if params.batch_idx_train<a.average_period:return
        rng=persistence.random_state(specaug)
        original={name:value.detach().cpu().clone() for name,value in net.state_dict().items()}
        saved_average=torch.load(a.output/f'step-{params.batch_idx_train:09d}.pt',map_location='cpu',mmap=True,weights_only=False)['model_avg']
        net.load_state_dict(saved_average,strict=True)
        del saved_average
        try:
            for split in ('development','test'):
                net.eval()
                result=metrics.evaluate(net,split,rank,world,a.output/'evaluations'/f'averaged-step-{params.batch_idx_train:09d}'/split/f'rank-{rank}',limit=a.eval_limit)
                if rank==0:
                    record=dict(optimizer_step=params.batch_idx_train,epoch=params.cur_epoch,checkpoint='native_cumulative_float64_average',**result)
                    with (a.output/'timit_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    for mode in ('text_free','fsa_text_free','forced_k2','mapped_arpabet','fsa_mapped_arpabet'):
                        tracking.run.log(dict(optimizer_step=params.batch_idx_train,**{'evaluation/averaged/'+split+'/'+mode+'/'+key:value for key,value in result[mode].items() if isinstance(value,(int,float))}))
        finally:
            net.load_state_dict(original,strict=True);persistence.restore_random(rng,specaug)

    original_validation=native.compute_validation_loss
    def validation(**kwargs):
        rng=persistence.random_state(specaug)
        try:
            val_batches=[[r['id']] for i,r in builtins.enumerate(validation_rows) if i%world==rank]
            valid_loader=Loader(valid_inputs,val_batches,0)
            result=original_validation(**dict(kwargs,valid_dl=valid_loader))
            if rank==0:
                with (a.output/'evaluation_loss.jsonl').open('a') as stream:stream.write(json.dumps(dict(optimizer_step=params.batch_idx_train,epoch=params.cur_epoch,**dict(result)))+'\n')
            benchmark('development')
            return result
        finally:persistence.restore_random(rng,specaug)

    native.compute_loss=compute_loss
    native.compute_validation_loss=validation
    native.save_checkpoint_with_global_batch_idx=lambda **kwargs:state.update(save_due=True)
    native.remove_checkpoints=lambda **kwargs:None
    original_plotter=native.ParamPlotter
    native.ParamPlotter=lambda model,tb_writer,period=50:original_plotter(TrainableView(model),tb_writer,period)
    # Native bad-model saving remains available; successful resumes use stronger
    # all-rank RNG and mixed-dtype optimizer persistence above.
    if not checkpoints:
        benchmark('development');benchmark('test')
    for epoch in range(progress['epoch'],a.epochs):
        if should_stop():
            tracking.finish();dist.destroy_process_group();return
        params.cur_epoch=epoch+1
        continuing=bool(checkpoints and loader.epoch==epoch and loader.cursor<len(loader.batches))
        if not continuing:
            native.fix_random_seed(42+epoch);torch.cuda.manual_seed(42+50*(epoch+1)+512*rank)
            specaug=native.AlternatingSpecAugment(seed=42+50*(epoch+1)+512*rank)
            loader=Loader(inputs,initial_plan if epoch==0 else plan(rows,a.max_duration,rank,world,epoch),epoch)
        loader.before_next=boundary;loader.stop=should_stop
        if not continuing:scheduler.set_epoch(epoch+1)
        if continuing:
            # Do not recompute the preceding epoch adjustment on same-epoch resume.
            scheduler.past_batches=sum(scheduler.batches_per_epoch[:epoch])
            scheduler._set_lrs()
        net.acoustic.encoder.compute_projection_overlap(verbose=True)
        native.train_one_epoch(params=params,model=model,optimizer=optimizer,scheduler=scheduler,
            sp=PhoneTokenizer(),train_dl=loader,valid_dl=None,scaler=scaler,model_avg=average,
            tb_writer=tracking if rank==0 else None,specaug=specaug,world_size=world,rank=rank)
        boundary()
        progress['epoch']=epoch if loader.cursor<len(loader.batches) else epoch+1
        save()
        if loader.cursor<len(loader.batches):
            tracking.finish();dist.destroy_process_group();return
        benchmark('development');benchmark('test')
        checkpoints=[]
    averaged_benchmark()
    if rank==0:(a.output/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(progress,completed_at=datetime.now(timezone.utc).isoformat()))+'\n')
    tracking.finish(completed=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
