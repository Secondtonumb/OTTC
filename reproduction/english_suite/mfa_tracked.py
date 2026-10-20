"""Instrument immutable MFA trainer logging and external dev; preserve its migration contract."""
import ast
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

SOURCE = Path('/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer/train_mfa_microbatch.py')


class Tracker:
    run = None
    output = None

    def log(self, record, output, arm):
        if int(os.environ.get('RANK', '0')) != 0:
            return
        import wandb
        if self.run is None:
            self.output = output
            identity_path = output/'wandb_run.json'
            identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
            config = json.loads((output/'run_config.json').read_text())
            if (output/'throughput_change.json').exists():
                config['microbatch_optimization'] = json.loads((output/'throughput_change.json').read_text())
            config.update({'logging_wrapper': str(Path(__file__).resolve()),
                           'logging_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'original_trainer_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                           'historical_backfill': False, 'job': os.environ.get('PBS_JOBID')})
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            torch_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            try:
                arguments = dict(project='ottc-mfa-english', entity='kevingenghaopeng-',
                                 id=identity.get('id'), resume='allow' if identity else None,
                                 name=output.name, group=os.environ.get('WANDB_RUN_GROUP', 'mfa_english_full'),
                                 config=config, dir=str(output), mode=os.environ.get('WANDB_MODE', 'online'),
                                 settings=wandb.Settings(init_timeout=45, disable_code=True))
                try:
                    self.run = wandb.init(**arguments)
                except wandb.errors.CommError:
                    arguments.update(mode='offline')
                    arguments.pop('resume', None)
                    self.run = wandb.init(**arguments)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng)
            self.run.config.update(config, allow_val_change=True)
            identity_path.write_text(json.dumps({'id': self.run.id, 'url': self.run.url})+'\n')
            self.run.define_metric('optimizer_step')
            self.run.define_metric('train/*', step_metric='optimizer_step')
            self.run.define_metric('dev/*', step_metric='optimizer_step')
        if record['event'] == 'update':
            payload = {'optimizer_step': record['step']}
            payload.update({'train/'+k: v for k, v in record.items() if k not in ['event', 'step', 'rank_ids'] and v is not None})
            self.run.log(payload)

    def development(self, net, arguments, progress, initial=False):
        from mfa_metrics import DATA, evaluate
        if not (DATA/'PREPARE_COMPLETE.json').exists():
            raise RuntimeError('External TIMIT development features must be ready before new MFA segments')
        python_rng,numpy_rng=random.getstate(),np.random.get_state()
        torch_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        was_training=net.training
        try:
            net.eval()
            result=evaluate(net,arguments.arm,rank=int(os.environ['RANK']),world=int(os.environ['WORLD_SIZE']))
        finally:
            net.train(was_training)
            random.setstate(python_rng);np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng);torch.cuda.set_rng_state(cuda_rng)
        if int(os.environ['RANK'])==0:
            record={'event':'development','optimizer_step':progress['global_step'],'epoch':progress['epoch'],
                    'initial_resume_evaluation':initial,**result}
            with (arguments.output/'dev_metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            (arguments.output/'latest_dev_metrics.json').write_text(json.dumps(record,indent=2)+'\n')
            print('DEVELOPMENT '+json.dumps(record),flush=True)
            if self.run is not None:
                payload={'optimizer_step':progress['global_step'],'dev/epoch':progress['epoch']}
                payload.update({'dev/'+k:v for k,v in result['text_free'].items() if isinstance(v,(int,float))})
                self.run.log(payload)
                self.run.summary.update({'latest/'+k:v for k,v in payload.items()})
            from summarize_results import main as summarize_results
            summarize_results()

    def progress(self, progress):
        if self.run is not None:
            self.run.summary.update({'progress/'+k: v for k, v in progress.items()})
            self.run.summary['training/complete'] = progress['epoch'] >= 10
            (self.output/'progress.json').write_text(json.dumps(progress)+'\n')

    def finish(self, code):
        if self.run is not None:
            self.run.finish(exit_code=code)


def main():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    trainer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    logger = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'log')
    logger.body.append(ast.parse('suite_tracker.log(record, a.output, a.arm)').body[0])
    checkpoint = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint')
    checkpoint.body.append(ast.parse('suite_tracker.progress(progress)').body[0])
    loop=next(n for n in trainer.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    position=trainer.body.index(loop)
    trainer.body.insert(position,ast.parse('suite_tracker.development(net, a, progress, initial=True)').body[0])
    trainer.body.insert(position,ast.parse("suite_tracker.log({'event':'initialization'}, a.output, a.arm)").body[0])
    # Each completed epoch evaluates the external dev split on all eight ranks.
    loop.body.append(ast.parse('suite_tracker.development(net, a, progress)').body[0])
    ast.fix_missing_locations(tree)
    tracker = Tracker()
    namespace = {'__file__': str(SOURCE), '__name__': 'mfa_tracked_source', 'suite_tracker': tracker}
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    try:
        namespace['main']()
    except BaseException:
        tracker.finish(1)
        raise
    tracker.finish(0)


if __name__ == '__main__':
    main()
