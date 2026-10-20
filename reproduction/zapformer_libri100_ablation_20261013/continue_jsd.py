"""Explicitly extend the accepted native recipe without editing its source/checkpoints."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'zapformer_s3t2ss_native_20261012'
sys.path.insert(0, str(NATIVE))
import runtime
import torch
import train_s3t2 as entry
from validate import source_identity
sys.path.insert(0, str(HERE))
from task_adapter import ConfigurableS3T2
from forward_sum import validate_config


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation-spec', type=Path, required=True)
    parser.add_argument('--loss-config',type=Path,default=HERE/'loss_config.json')
    for key in ('ottc_weight','forward_sum_weight','forward_sum_blank_logprob','forward_sum_temperature','forward_sum_output_beam'):
        parser.add_argument('--'+key.replace('_','-'),type=float)
    for key in ('ottc_cost','forward_sum_kind','forward_sum_normalization','forward_sum_reduction','forward_sum_ctc_device'):
        parser.add_argument('--'+key.replace('_','-'))
    parser.add_argument('--forward-sum-zero-infinity',action=argparse.BooleanOptionalAction,default=None)
    args, native_args = parser.parse_known_args()
    spec = json.loads(args.continuation_spec.read_text())
    config=json.loads(args.loss_config.read_text())
    for key in list(config):
        value=getattr(args,key,None)
        if value is not None:config[key]=value
    config=validate_config(config)
    assert config==spec['loss_config'],'Loss overrides must match the saved experiment specification'
    entry.S3T2=lambda *pos,**kw:ConfigurableS3T2(*pos,loss_config=config,**kw)
    assert source_identity() == spec['source_identity'], 'Accepted native source changed'
    assert sha(spec['origin_checkpoint']) == spec['origin_checkpoint_sha256']
    assert spec['scheduler_policy'] in ('native_replanned_total', 'constant_floor')
    origin = torch.load(spec['origin_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    origin_contract = copy.deepcopy(origin['contract'])
    assert origin_contract['objective'] == 'jsd' and origin_contract['ottc_cost'] == 'jsd_upstream'
    assert origin_contract['topology'] == 's3_t2_star_star'
    assert origin['progress']['epoch'] == origin_contract['epochs']
    del origin
    extension = dict(
        origin_checkpoint=spec['origin_checkpoint'],
        origin_checkpoint_sha256=spec['origin_checkpoint_sha256'],
        origin_epochs=origin_contract['epochs'], target_epochs=spec['target_epochs'],
        scheduler_policy=spec['scheduler_policy'],
        wrapper_sha256=sha(__file__),
        loss_config=config,
        loss_adapter_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        charsiu_source=json.loads((HERE/'charsiu_source.json').read_text()),
        score_source='Existing 313-state acoustic output; logsumexp over three states per phone; transcript-position gather and normalization; no text encoder',
        bookkeeping='New objective branch resets best-loss reporting; same-epoch loss accumulator persisted for deterministic reporting on resumes',
        history='Completed original schedule retained; continuation extends actual history, not a fresh 50-epoch reproduction',
    )
    original_load = entry.persistence.load
    epoch_metrics={}
    pending_metrics={}

    def load(path, model, average, optimizer, scheduler, scaler, specaug, sampler, contract):
        assert contract['epochs'] == spec['target_epochs'] > origin_contract['epochs']
        expected = copy.deepcopy(origin_contract)
        expected['epochs'] = spec['target_epochs']
        expected['epoch_batches'] = origin_contract['epoch_batches'] + [origin_contract['epoch_batches'][-1]] * (spec['target_epochs'] - origin_contract['epochs'])
        assert contract == expected, 'Unexpected recipe change beyond requested epoch extension'
        # Native entry's --objective jsd selects the IPA sequence-data route.
        # Record the actual configurable objective in the saved contract.
        contract['objective']='jsd' if config['ottc_cost']=='jsd_upstream' else 'ottc_ce'
        contract['ottc_cost']=config['ottc_cost']
        contract['continuation'] = extension
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        saved_contract = copy.deepcopy(state['contract'])
        first_extension = saved_contract == origin_contract
        if first_extension:
            assert sha(path) == spec['origin_checkpoint_sha256'], 'Only the specified completed checkpoint may extend its contract'
        else:
            assert saved_contract == contract, 'Continuation contract changed'
        del state
        if spec['scheduler_policy'] == 'constant_floor':
            scheduler.get_lr = lambda: [lr * scheduler.min_factor for lr in scheduler.base_lrs]
        # Native persistence restores optimizer/scaler/RNG/averages exactly.
        # Its scheduler does not persist the horizon: construction uses the new
        # batches_per_epoch; only epoch/batch/adjust_factor are restored.
        progress = original_load(path, model, average, optimizer, scheduler, scaler, specaug, sampler, saved_contract)
        if first_extension:
            progress['native_statistics']=dict(best_train_epoch=0,best_valid_epoch=0,best_train_loss=float('inf'),best_valid_loss=float('inf'))
        else:
            rank=entry.dist.get_rank()
            state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            pending_metrics.update(state['rank_states'][rank]['sampler'].get('native_epoch_tot_loss',{}))
            del state
        assert progress['epoch'] >= origin_contract['epochs']
        print(json.dumps(dict(continuation=extension, restored_global_step=progress['global_step'],
                              restored_epoch=progress['epoch'], replanned_lr=max(scheduler.get_last_lr()))), flush=True)
        return progress

    entry.persistence.load = load
    original_next=entry.Loader.__next__
    def loader_next(self):
        frame=inspect.currentframe().f_back
        if frame.f_code.co_name=='train_one_epoch':
            epoch_metrics.clear();epoch_metrics.update(dict(frame.f_locals['tot_loss']))
        return original_next(self)
    entry.Loader.__next__=loader_next
    original_save=entry.persistence.save
    def save(path,model,average,optimizer,scheduler,scaler,specaug,sampler,progress,contract):
        sampler=dict(sampler,native_epoch_tot_loss=dict(epoch_metrics))
        return original_save(path,model,average,optimizer,scheduler,scaler,specaug,sampler,progress,contract)
    entry.persistence.save=save
    original_train=entry.native.train_one_epoch
    metrics_type=entry.native.MetricsTracker
    def train_one_epoch(**kwargs):
        params=kwargs['params'];loader=kwargs['train_dl']
        first=[True]
        continuing=loader.cursor>0 and loader.cursor<len(loader.batches)
        def metrics_factory(*pos,**kw):
            result=metrics_type(*pos,**kw)
            if first[0]:
                first[0]=False
                if continuing:result.update(pending_metrics)
                pending_metrics.clear()
            return result
        saved_stats=(params.best_train_epoch,params.best_train_loss)
        entry.native.MetricsTracker=metrics_factory
        try:original_train(**kwargs)
        finally:entry.native.MetricsTracker=metrics_type
        if loader.cursor<len(loader.batches):
            params.best_train_epoch,params.best_train_loss=saved_stats
    entry.native.train_one_epoch=train_one_epoch
    original_tracking = entry.Tracking.__init__

    def tracking_init(self, output, contract, rank, smoke=False):
        # Resume existing W&B configuration before explicitly recording the
        # extended plan; no original loss points are re-logged or interpolated.
        identity = output / 'wandb_run.json'
        original_tracking(self, output, contract, rank, smoke)
        if self.run:
            self.run.config.update(contract, allow_val_change=True)
            self.run.summary['training/complete'] = False
            self.run.summary['continuation/origin_epochs'] = origin_contract['epochs']
            self.run.summary['continuation/target_epochs'] = spec['target_epochs']
            self.run.summary['continuation/parent_wandb_url']=spec['parent_wandb_url']

    # The native entry loads its scorer by importlib; annotate the retained
    # S3T2** evaluation graph explicitly when the auxiliary training graph differs.
    original_spec=entry.importlib.util.spec_from_file_location
    def metric_spec(name,path,*pos,**kw):
        if name=='native_topology_metrics':path=HERE/'metrics_adapter.py'
        return original_spec(name,path,*pos,**kw)
    entry.importlib.util.spec_from_file_location=metric_spec

    entry.Tracking.__init__ = tracking_init
    sys.argv = [str(NATIVE / 'train_s3t2.py')] + native_args
    entry.main()


if __name__ == '__main__':
    main()
