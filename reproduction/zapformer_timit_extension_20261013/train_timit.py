"""Single-GPU TIMIT monophone OTTC using the archived native Zapformer loop."""
import bootstrap
from bootstrap import HERE, ROOT, NATIVE
import argparse
import builtins
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import train as native
import checkpoint as persistence
from task_model import parameters
from train_s3t2 import PhoneTokenizer
from data import Inputs, Loader
from timit_model import TimitOTTC
from evaluate import evaluate
from extension_schedule import continuation_scheduler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batches(rows, max_duration, epoch, seed):
    """Padded-duration budget, deterministic length buckets, no dropped rows."""
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(len(rows)).tolist()
    result = []
    for offset in range(0,len(order),256):
        bucket = sorted(order[offset:offset+256], key=lambda i: rows[i]['frames'])
        current=[]
        for i in bucket:
            if rows[i]['frames']*.01 > max_duration:
                raise ValueError('One utterance exceeds max-duration; increase budget')
            if current and rows[i]['frames']*.01*(len(current)+1)>max_duration:
                result.append([rows[k]['id'] for k in current]); current=[]
            current.append(i)
        if current: result.append([rows[k]['id'] for k in current])
    rng.shuffle(result)
    assert sum(map(len,result)) == len(rows)
    return result


class Log:
    def __init__(self, output, contract, mode):
        import wandb
        self.output=output
        defaults=json.loads((ROOT/'reproduction/wandb_config.json').read_text())
        identity=output/'wandb_run.json'
        saved=json.loads(identity.read_text()) if identity.exists() else {}
        for key,name in [('WANDB_DATA_DIR','wandb_data'),('WANDB_CACHE_DIR','wandb_cache')]:
            path=output/name; path.mkdir(exist_ok=True)
            os.environ.setdefault(key,str(path.resolve()))
        options=dict(project=defaults['project'],entity=defaults.get('entity'),
                     name=output.name,group='zapformer_timit_monophone_extension_20261013',
                     dir=str(output.resolve()),config=contract,mode=mode,
                     id=saved.get('id'),resume='allow' if saved else None,
                     settings=wandb.Settings(init_timeout=35))
        try:self.run=wandb.init(**options)
        except (wandb.errors.CommError,wandb.errors.UsageError):
            if mode!='online':raise
            options.update(mode='offline');options.pop('resume',None)
            self.run=wandb.init(**options)
        identity.write_text(json.dumps(dict(id=self.run.id,mode=self.run.settings.mode))+'\n')
        self.run.define_metric('step');self.run.define_metric('*',step_metric='step')

    def record(self, record):
        with (self.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        self.run.log(record)
        print(json.dumps(record),flush=True)

    def finish(self, complete):
        from reproduction.wandb_policy import log_evaluation_artifact
        import wandb
        self.run.summary['complete']=complete
        artifact=wandb.Artifact(self.output.name+'-metrics',type='evaluation')
        for p in [self.output/'run_config.json',self.output/'history.jsonl',*self.output.glob('final_*.json')]:
            if p.exists():artifact.add_file(str(p))
        log_evaluation_artifact(self.run, artifact);self.run.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'reproduction/zapformer_timit_20261013/data')
    p.add_argument('--parent-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cost',choices=['jsd','ce'],default='jsd')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--base-lr',type=float,default=.02)
    p.add_argument('--max-duration',type=float,default=100.)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--average-period',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-wall-seconds',type=float,default=10200)
    p.add_argument('--stop-after',type=int,default=0,help='Absolute step, for validation/resume')
    p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--eval-limit',type=int,default=0)
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default=os.environ.get('WANDB_MODE','online'))
    a=p.parse_args()
    a.parent_output=a.parent_output.resolve();a.output=a.output.resolve()
    if a.output==a.parent_output:p.error('Extension needs a separate output directory')
    parent_contract=json.loads((a.parent_output/'run_config.json').read_text())
    parent_complete=a.parent_output/'TRAIN_COMPLETE.json'
    if not parent_complete.exists():p.error('Finish the original run before starting its extension')
    parent_progress=json.loads(parent_complete.read_text())
    if parent_progress['epoch']!=parent_contract['epochs']:p.error('Parent is not complete')
    if a.epochs<=parent_contract['epochs']:p.error('Extension total epochs must exceed parent epochs')
    if not (a.parent_output/'resume.pt').is_file() or not (a.parent_output/'best.pt').is_file():
        p.error('Parent needs both the full resumable checkpoint and best checkpoint')
    for filename,digest in parent_contract['source_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Parent source changed: '+filename)
        if Path(filename).name in ('bootstrap.py','timit_model.py','evaluate.py'):
            if sha(HERE/Path(filename).name)!=digest:
                raise ValueError('Copied model/evaluator differs from parent: '+filename)
    if not torch.cuda.is_available():p.error('Run this entry on an interactive GPU node')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:p.error('This entry is single-GPU only')
    if min(a.epochs,a.save_every,a.average_period,a.base_lr,a.max_duration)<=0:p.error('Positive training parameters required')
    if min(a.train_limit,a.eval_limit,a.stop_after,a.max_wall_seconds)<0:p.error('Limits must be nonnegative')
    torch.cuda.set_device(0);torch.set_num_threads(4)
    native.fix_random_seed(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_multithreading_enabled(False)
    metadata=json.loads((a.data/'READY.json').read_text())
    rows={}
    for split in ('train','dev','test'):
        path=a.data/f'{split}.jsonl'
        if sha(path)!=metadata['splits'][split]['rows_sha256']:raise ValueError('Changed cached rows')
        if sha(a.data/f'{split}.f32')!=metadata['splits'][split]['feature_sha256']:raise ValueError('Changed cached features')
        rows[split]=[json.loads(x) for x in path.read_text().splitlines()]
        limit=a.train_limit if split=='train' else a.eval_limit
        if limit: rows[split]=rows[split][:limit]
    source_manifest=json.loads((NATIVE/'vendor_manifest.json').read_text())
    for name,digest in source_manifest['sha256'].items():
        if sha(NATIVE/'vendor'/name)!=digest:raise ValueError('Native source changed: '+name)
    plans=[batches(rows['train'],a.max_duration,e,a.seed) for e in range(a.epochs)]
    contract=dict(task='TIMIT40 monophone OTTC',cost=a.cost,upstream_jsd=a.cost=='jsd',ctc_loss=False,forward_sum=False,
                  frame_ms=10,initialization=f'random seed{a.seed}',backbone='full Zapformer' if not a.tiny else 'tiny validation Zapformer',
                  epochs=a.epochs,base_lr=a.base_lr,max_padded_duration=a.max_duration,seed=a.seed,
                  average_period=a.average_period,save_every=a.save_every,
                  train_count=len(rows['train']),dev_count=len(rows['dev']),test_count=len(rows['test']),
                  data_sha256=sha(a.data/'READY.json'),tiny=a.tiny,
                  train_limit=a.train_limit,eval_limit=a.eval_limit,
                  optimizer='native BatchedRubik beta1=.99',scheduler='native InterpCosine min_factor=.025 linear_scale=.5',
                  precision='BF16 + native GradScaler',native_train_loop=True,
                  aux_loss_scale=.05,aux_after_step2000=.25,loss_reduction='sum utterance transport costs',
                  augmentation='native AlternatingSpecAugment',sampling='padded-duration length buckets; every train utterance once per epoch; no speed perturbation',
                  source_sha256={str(path):sha(path) for path in [*HERE.glob('*.py'),NATIVE/'archived_ot_loss.py',NATIVE/'checkpoint.py',NATIVE/'augmentation.py',NATIVE/'task_model.py',NATIVE/'data.py',Path(native.__file__)]},
                  native_vendor_manifest_sha256=sha(NATIVE/'vendor_manifest.json'),
                  timit_gradient_training=True,selection='minimum dev TIMIT40 raw PER; test only at completion')
    ignored={'epochs','source_sha256'}
    changed=[key for key,value in parent_contract.items() if key not in ignored and contract.get(key)!=value]
    if changed:raise ValueError('Extension changed parent recipe fields: '+', '.join(changed))
    contract['parent']=dict(output=str(a.parent_output),epochs=parent_contract['epochs'],
        global_step=parent_progress['global_step'],
        run_config_sha256=sha(a.parent_output/'run_config.json'),
        completion_sha256=sha(parent_complete),
        wandb_run_id=json.loads((a.parent_output/'wandb_run.json').read_text()).get('id'))
    contract['scheduler']='native InterpCosine through parent horizon; constant final group LRs thereafter'
    contract['extension_lr_policy']='no LR restart; base .02 x min_factor .025 = .0005 before native group scaling'
    a.output.mkdir(parents=True,exist_ok=True)
    saved_path=a.output/'resume.pt'
    if (a.output/'run_config.json').exists() and json.loads((a.output/'run_config.json').read_text())!=contract:
        raise ValueError('Resume contract changed; choose a new output')
    if (a.output/'TRAIN_COMPLETE.json').exists():
        print('Already complete:',a.output);return
    if (a.output/'run_config.json').exists() and not saved_path.exists():
        raise ValueError('Existing run has no resumable checkpoint; choose a new output')
    net=TimitOTTC(a.cost,a.tiny)
    average=copy.deepcopy(net).double()
    net.cuda()
    groups=native.get_parameter_groups_with_lrs(net,lr=a.base_lr,include_names=False)
    optimizer=native.BatchedRubik(groups,lr=a.base_lr,beta1=.99)
    Scheduler=continuation_scheduler(native.InterpCosineLRScheduler,parent_contract['epochs'])
    scheduler=Scheduler(optimizer,batches_per_epoch=[len(x) for x in plans],min_factor=.025,linear_scale=.5)
    scaler=native.GradScaler(enabled=True,init_scale=1.)
    params=parameters(a.tiny)
    params.update(exp_dir=a.output,cur_epoch=1,world_size=1,max_duration=a.max_duration,
                  use_autocast=True,dtype=torch.bfloat16,aux_loss_scale=.05,
                  average_period=a.average_period,save_every_n=10**9,keep_last_k=0,
                  log_interval=50,valid_interval=10**9,print_diagnostics=False,inf_check=False,subsampling_factor=1)
    progress=dict(epoch=0,global_step=0,best_dev_per=math.inf,native_statistics={})
    epoch=0
    load_path=saved_path if saved_path.exists() else a.parent_output/'resume.pt'
    load_contract=contract if saved_path.exists() else parent_contract
    if load_path.exists():
        saved=torch.load(load_path,map_location='cpu',mmap=True,weights_only=False)
        if saved['contract']!=load_contract:raise ValueError('Checkpoint contract mismatch')
        if load_path!=saved_path and saved['progress']['global_step']!=parent_progress['global_step']:
            raise ValueError('Parent checkpoint no longer matches completion marker')
        epoch=saved['rank_states'][0]['sampler']['epoch'];del saved
    inputs=Inputs(rows['train'])
    loader=Loader(inputs,plans[epoch],epoch)
    specaug=native.AlternatingSpecAugment(seed=a.seed+50*(epoch+1))
    if load_path.exists():
        progress=persistence.load(load_path,net,average,optimizer,scheduler,scaler,specaug,loader,load_contract)
        params.batch_idx_train=progress['global_step']
        params.update(progress['native_statistics'])
        if load_path!=saved_path:
            best=torch.load(a.parent_output/'best.pt',map_location='cpu',weights_only=False)
            if best['contract']!=parent_contract:raise ValueError('Parent best checkpoint contract mismatch')
            best['contract']=contract
            torch.save(best,a.output/'best.pt.partial')
            os.replace(a.output/'best.pt.partial',a.output/'best.pt')
            del best
    else:
        scheduler.set_epoch(1);scheduler.set_batch(0)
    rng=persistence.random_state(specaug)
    log=Log(a.output,contract,a.wandb_mode)
    persistence.restore_random(rng,specaug)
    (a.output/'run_config.json').write_text(json.dumps(contract,indent=2)+'\n')
    state=dict(last=None,stop=False,last_saved=-1,start=time.monotonic())
    # Ctrl-C / TERM requests a checkpoint at the next complete optimizer boundary.
    def stop_signal(*_):state['stop']=True
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    def should_stop():
        return state['stop'] or (a.stop_after and params.batch_idx_train>=a.stop_after) or (a.max_wall_seconds and time.monotonic()-state['start']>=a.max_wall_seconds)
    def save():
        progress.update(global_step=params.batch_idx_train,native_statistics={k:params[k] for k in ('best_train_epoch','best_train_loss','best_valid_epoch','best_valid_loss')})
        persistence.save(saved_path,net,average,optimizer,scheduler,scaler,specaug,loader.state_dict(),progress,contract)
        state['last_saved']=params.batch_idx_train
        (a.output/'progress.json').write_text(json.dumps(progress)+'\n')
    def flush():
        if state['last'] is not None:
            log.record(dict(step=params.batch_idx_train,epoch=params.cur_epoch,lr=max(scheduler.get_last_lr()),grad_scale=scaler.get_scale(),**state['last']))
            state['last']=None
        if params.batch_idx_train%a.save_every==0 and state['last_saved']!=params.batch_idx_train:save()
    # Commit a complete, resumable inherited state before the first extension update.
    # No historical loss points are copied to this independent continuation run.
    if not saved_path.exists():
        save()
        log.record(dict(step=params.batch_idx_train,epoch=progress['epoch'],inherited_checkpoint=True))
    def compute_loss(params,model,sp,batch,is_training,aux_loss_scale=0.,specaug=None):
        features=batch['inputs'].cuda();lengths=batch['supervisions']['num_frames'].cuda()
        if specaug is not None:
            with torch.autocast('cuda',enabled=False):features=specaug(features.float(),lengths)
        loss=model(features,lengths,batch['tokens'].cuda(),batch['token_lengths'].cuda(),aux_loss_scale)
        info=native.MetricsTracker();info['frames']=int(lengths.sum());info['loss']=float(loss.detach())
        state['last']=dict(train_loss_sum=info['loss'],train_loss_per_utterance=info['loss']/len(lengths),frames=info['frames'])
        return loss,info
    def validation(**kwargs):
        # Native loop calls this on the first batch of an epoch. Keep actual
        # full-dev loss/PER (never replace this with fabricated zero validation).
        result=evaluate(net,rows['dev'],metadata['contract']['vocab'])
        log.record(dict(step=params.batch_idx_train,epoch=params.cur_epoch,dev_periodic_loss=result['loss_per_utterance'],dev_periodic_PER=result['raw_PER_percent']))
        info=native.MetricsTracker();info['frames']=1;info['loss']=result['loss_per_utterance']
        return info
    native.compute_loss=compute_loss;native.compute_validation_loss=validation
    native.enumerate=lambda iterable,start=0:builtins.enumerate(iterable,start=getattr(iterable,'batch_offset',start))
    complete=False
    try:
        for epoch in range(progress['epoch'],a.epochs):
            params.cur_epoch=epoch+1
            continuing=loader.epoch==epoch and 0<loader.cursor<len(loader.batches)
            if not continuing:
                native.fix_random_seed(a.seed+epoch)
                specaug=native.AlternatingSpecAugment(seed=a.seed+50*(epoch+1))
                loader=Loader(inputs,plans[epoch],epoch)
                scheduler.set_epoch(epoch+1)
            progress['epoch']=epoch
            loader.before_next=flush;loader.stop=should_stop
            if not should_stop():
                net.acoustic.encoder.compute_projection_overlap(verbose=False)
                native.train_one_epoch(params=params,model=net,optimizer=optimizer,scheduler=scheduler,
                    sp=PhoneTokenizer(),train_dl=loader,valid_dl=None,scaler=scaler,model_avg=average,
                    specaug=specaug,world_size=1,rank=0)
            flush()
            if loader.cursor<len(loader.batches):save();return
            result=evaluate(net,rows['dev'],metadata['contract']['vocab'],output=a.output/f'dev_epoch{epoch+1:03d}.json')
            log.record(dict(step=params.batch_idx_train,epoch=epoch+1,dev_PER=result['raw_PER_percent'],dev_loss=result['loss_per_utterance'],
                            dev_F1_20ms=result['text_free']['20']['f1_pct'],dev_R_20ms=result['text_free']['20']['r_value_pct'],
                            dev_precision_20ms=result['text_free']['20']['precision_pct'],
                            dev_recall_20ms=result['text_free']['20']['recall_pct']))
            if result['raw_PER_percent']<progress['best_dev_per']:
                progress['best_dev_per']=result['raw_PER_percent']
                torch.save(dict(model=net.state_dict(),contract=contract,epoch=epoch+1,step=params.batch_idx_train),a.output/'best.pt.partial')
                os.replace(a.output/'best.pt.partial',a.output/'best.pt')
            progress['epoch']=epoch+1;save()
        best=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(best['model'])
        for split in ('dev','test'):
            result=evaluate(net,rows[split],metadata['contract']['vocab'],forced=True,output=a.output/f'final_{split}.json')
            log.run.summary.update({f'final/{split}/PER':result['raw_PER_percent'],f'final/{split}/FA_MAE_ms':result['forced_k2_mae_ms'],
                                    f'final/{split}/FA_F1_20ms':result['forced_k2']['20']['f1_pct'],f'final/{split}/FA_R_20ms':result['forced_k2']['20']['r_value_pct']})
            for mode in ('text_free','forced_k2'):
                for metric in ('precision_pct','recall_pct','f1_pct','r_value_pct'):
                    log.run.summary[f'final/{split}/{mode}/{metric}_20ms']=result[mode]['20'][metric]
        complete=True
        (a.output/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(progress,selected_epoch=best['epoch'],selected_step=best['step']))+'\n')
    finally:
        log.finish(complete)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()
