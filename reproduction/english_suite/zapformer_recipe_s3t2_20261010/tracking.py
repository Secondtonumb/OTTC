"""Independent W&B identity plus full original TIMIT development metrics."""
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import checkpoint

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent


class Tracker:
    def __init__(self,output,contract,rank):
        self.output=output;self.rank=rank;self.run=None
        if rank:return
        import wandb
        defaults=json.loads((SUITE.parent/'wandb_config.json').read_text())
        identity_path=output/'wandb_run.json'
        identity=json.loads(identity_path.read_text()) if identity_path.exists() else {}
        # Logging initialization may consume Python/Torch RNG; restore it immediately.
        py,np_rng=random.getstate(),np.random.get_state()
        cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        args=dict(project=os.environ.get('WANDB_PROJECT',defaults['project']),
                  entity=os.environ.get('WANDB_ENTITY',defaults.get('entity')),
                  mode=os.environ.get('WANDB_MODE',defaults.get('mode','online')),
                  id=identity.get('id'),resume='allow' if identity else None,
                  name='zapformer_spottc_jsd_s3t2_recipe_scratch',group='zapformer_recipe_s3t2_20261010',
                  config=dict(contract=contract,job=os.environ.get('PBS_JOBID'),historical_backfill=False),
                  dir=str(output),settings=wandb.Settings(init_timeout=45,disable_code=True))
        try:
            try:self.run=wandb.init(**args)
            except wandb.errors.CommError:
                args.update(mode='offline');args.pop('resume',None);self.run=wandb.init(**args)
            identity_path.write_text(json.dumps({'id':self.run.id,'url':self.run.url})+'\n')
            self.run.define_metric('optimizer_step')
            for prefix in ['train/*','dev/*','final/*']:
                self.run.define_metric(prefix,step_metric='optimizer_step')
        finally:
            random.setstate(py);np.random.set_state(np_rng);torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)

    def log(self,record):
        if self.run and record['event']=='update':
            self.run.log(dict(optimizer_step=record['step'],**{'train/'+k:v for k,v in record.items()
                             if k not in ['event','step','rank_ids'] and isinstance(v,(int,float))}))

    def development(self,net,progress):
        sys.path.append(str(SUITE))
        from mfa_metrics import evaluate
        py,np_rng=random.getstate(),np.random.get_state();cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        was_training=net.training
        try:
            net.eval()
            result=evaluate(net,'spottc',rank=self.rank,world=torch.distributed.get_world_size())
        finally:
            net.train(was_training);random.setstate(py);np.random.set_state(np_rng)
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        if not self.rank:
            record=dict(event='development',optimizer_step=progress['global_step'],epoch=progress['epoch'],**result)
            with (self.output/'dev_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            (self.output/'latest_dev_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
            if self.run:self.run.log(dict(optimizer_step=progress['global_step'],**{'dev/'+k:v for k,v in result['text_free'].items() if isinstance(v,(int,float))}))

    def finish(self):
        if self.run:self.run.finish()
