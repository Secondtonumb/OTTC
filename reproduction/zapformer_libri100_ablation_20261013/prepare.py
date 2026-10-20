"""Create a continuation output; preserve the completed ten-epoch run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from continue_jsd import NATIVE, sha, source_identity, torch
from forward_sum import validate_config

ORIGIN = HERE.parent / 'zapformer_libri100_s3t2ss_native_20261012/training'


def prepare(output, validation=False, arm=None):
    output = Path(output)
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    complete = json.loads((ORIGIN / 'TRAIN_COMPLETE.json').read_text())
    assert complete['epoch'] == 10 and complete['global_step'] == 1060
    checkpoint = ORIGIN / 'step-000001060.pt'
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert state['world_size'] == len(state['rank_states']) == 8
    assert state['progress']['epoch'] == 10 and state['progress']['global_step'] == 1060
    assert state['contract'] == json.loads((ORIGIN / 'run_config.json').read_text())
    assert state['contract']['adapter_sha256'] == {p.name:sha(p) for p in NATIVE.glob('*.py')}
    del state
    config=validate_config(json.loads((HERE/'configs'/f'{arm}.json').read_text()))
    spec = dict(origin_checkpoint=str(checkpoint.resolve()), origin_checkpoint_sha256=sha(checkpoint),
                source_identity=source_identity(), target_epochs=50, scheduler_policy='native_replanned_total',
                loss_config=config,arm=arm,
                parent_wandb_url=json.loads((ORIGIN/'wandb_run.json').read_text())['url'])
    output.mkdir(parents=True)
    # The trainer writes new names atomically and never mutates loaded weights.
    # Removing this continuation link during retention leaves the original intact.
    os.link(checkpoint, output / checkpoint.name)
    ancestor=output/'ancestor_metrics';ancestor.mkdir()
    for name in ('updates.jsonl', 'evaluation_loss.jsonl', 'timit_metrics.jsonl', 'progress.json', 'run_config.json','wandb_run.json'):
        shutil.copyfile(ORIGIN / name, ancestor / name)
    if validation:
        (output / 'wandb_run.json').write_text(json.dumps(dict(id=uuid.uuid4().hex[:8], mode='offline'))+'\n')
    (output / 'continuation_spec.json').write_text(json.dumps(spec, indent=2)+'\n')
    (output/'loss_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output / 'BOOTSTRAP.json').write_text(json.dumps(dict(
        origin_training=str(ORIGIN.resolve()), origin_checkpoint=spec['origin_checkpoint'],
        origin_checkpoint_sha256=spec['origin_checkpoint_sha256'],
        original_completion_preserved=True, inherited_optimizer_steps=1060,
        inherited_epochs=10, target_total_epochs=50, world_size=8,
        scheduler_policy=spec['scheduler_policy'], parent_wandb_url=spec['parent_wandb_url'],
        wandb_identity='New loss branch gets its own run; resumes reuse that saved ID',
        loss_config=spec['loss_config']), indent=2)+'\n')
    print(json.dumps(dict(output=str(output), target_epochs=50, preserved_checkpoint=spec['origin_checkpoint'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--arm',choices=('jsd_only','ce_fs','ce_only'),required=True)
    args=parser.parse_args()
    prepare(args.output, args.validation, args.arm)
