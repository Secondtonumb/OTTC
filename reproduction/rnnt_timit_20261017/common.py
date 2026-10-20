"""Local records, W&B identity and atomic checkpoints for the new recipe."""
import hashlib
import json
import os
import random

import numpy as np
import torch

from reproduction.cif_timit_20261016.data import ROOT


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        torch.cuda.set_rng_state_all(value['cuda'])


class Tracking:
    def __init__(self, output, contract, mode):
        self.output, self.run = output, None
        if mode == 'disabled':
            return
        import wandb
        defaults = json.loads((ROOT/'reproduction/wandb_config.json').read_text())
        path = output/'wandb_run.json'
        saved = json.loads(path.read_text()) if path.exists() else {}
        project = os.environ.get('WANDB_PROJECT', defaults['project'])
        entity = os.environ.get('WANDB_ENTITY', defaults.get('entity'))
        if saved and (saved['project'], saved['entity']) != (project, entity):
            raise ValueError('resume must retain W&B project/entity')
        # Save ID BEFORE initialization so a failed/interrupting init can resume.
        run_id = saved.get('id') or wandb.util.generate_id()
        write_json(path, dict(id=run_id, project=project, entity=entity, origin='live', mode=mode))
        options = dict(project=project, entity=entity, id=run_id, name=output.name,
            group='rnnt_timit40_20261017', dir=str(output.resolve()), config=contract,
            mode=mode, settings=wandb.Settings(init_timeout=35))
        if saved and mode == 'online':
            options['resume'] = 'allow'
        try:
            self.run = wandb.init(**options)
        except wandb.errors.CommError:
            if mode != 'online':
                raise
            options.update(mode='offline')
            options.pop('resume', None)
            self.run = wandb.init(**options)
        write_json(path, dict(id=run_id, project=project, entity=entity, origin='live',
            mode=self.run.settings.mode))
        self.run.define_metric('step')
        self.run.define_metric('*', step_metric='step')

    def log(self, values):
        line = json.dumps(values, allow_nan=False)
        with (self.output/'history.jsonl').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
        if self.run is not None:
            self.run.log(values)

    def finish(self, complete=False, failed=False):
        from reproduction.wandb_policy import log_evaluation_artifact
        if self.run is None:
            return
        import wandb
        self.run.summary['training_complete'] = complete
        artifact = wandb.Artifact(self.output.name + '-summaries', type='evaluation')
        for path in [self.output/'run_config.json', self.output/'history.jsonl',
                self.output/'checkpoint_manifest.json', *self.output.glob('*.json')]:
            if path.exists() and path.name not in ('wandb_run.json', 'config.json'):
                artifact.add_file(str(path))
        log_evaluation_artifact(self.run, artifact)
        self.run.finish(exit_code=int(failed))
