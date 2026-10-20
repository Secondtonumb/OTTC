"""Explicitly extend the accepted native recipe without editing its source/checkpoints."""
import argparse
import copy
import hashlib
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation-spec', type=Path, required=True)
    args, native_args = parser.parse_known_args()
    spec = json.loads(args.continuation_spec.read_text())
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
        history='Completed original schedule retained; continuation extends actual history, not a fresh 50-epoch reproduction',
    )
    original_load = entry.persistence.load

    def load(path, model, average, optimizer, scheduler, scaler, specaug, sampler, contract):
        assert contract['epochs'] == spec['target_epochs'] > origin_contract['epochs']
        expected = copy.deepcopy(origin_contract)
        expected['epochs'] = spec['target_epochs']
        expected['epoch_batches'] = origin_contract['epoch_batches'] + [origin_contract['epoch_batches'][-1]] * (spec['target_epochs'] - origin_contract['epochs'])
        assert contract == expected, 'Unexpected recipe change beyond requested epoch extension'
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
        assert progress['epoch'] >= origin_contract['epochs']
        print(json.dumps(dict(continuation=extension, restored_global_step=progress['global_step'],
                              restored_epoch=progress['epoch'], replanned_lr=max(scheduler.get_last_lr()))), flush=True)
        return progress

    entry.persistence.load = load
    original_tracking = entry.Tracking.__init__

    def tracking_init(self, output, contract, rank, smoke=False):
        # Resume existing W&B configuration before explicitly recording the
        # extended plan; no original loss points are re-logged or interpolated.
        identity = output / 'wandb_run.json'
        original_tracking(self, output, origin_contract if identity.exists() else contract, rank, smoke)
        if self.run:
            self.run.config.update(contract, allow_val_change=True)
            self.run.summary['training/complete'] = False
            self.run.summary['continuation/origin_epochs'] = origin_contract['epochs']
            self.run.summary['continuation/target_epochs'] = spec['target_epochs']

    entry.Tracking.__init__ = tracking_init
    sys.argv = [str(NATIVE / 'train_s3t2.py')] + native_args
    entry.main()


if __name__ == '__main__':
    main()
