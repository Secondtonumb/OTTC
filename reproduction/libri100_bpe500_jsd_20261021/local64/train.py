"""Continue global64 experiments on one GPU with physical local batch64."""
import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.libri100_bpe500_jsd_20261021 import train as old
from reproduction.libri100_bpe500_jsd_20261021.local64 import prepare
from transformers import Trainer, TrainingArguments, PretrainedConfig
from reproduction.checkpoint_retention import CheckpointRetentionCallback

parent, base, torch = old.parent, old.base, old.torch
dist = old.dist
ORIGINAL_SOURCES = old.source_hashes()


def source_hashes():
    sources = dict(ORIGINAL_SOURCES)
    for path in [*HERE.glob('*.py'), *HERE.glob('*.pbs')]:
        sources[str(path.relative_to(ROOT))] = old.prepare.sha256(path)
    return sources


class SingleTrainer(parent.CompleteCheckpoint, Trainer):
    compute_loss = old.DistributedTrainer.compute_loss
    prediction_step = old.DistributedTrainer.prediction_step

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiled_steps = 0

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        saved = torch.load(Path(checkpoint) / 'rng_state.pth', map_location='cpu', weights_only=False)
        random.setstate(saved['python'])
        base.np.random.set_state(saved['numpy'])
        torch.random.set_rng_state(saved['cpu'])
        if torch.cuda.is_available() and 'cuda' in saved:
            if torch.is_tensor(saved['cuda']):
                torch.cuda.set_rng_state(saved['cuda'], device=self.args.device)
            else:
                torch.cuda.set_rng_state_all(saved['cuda'])

    def training_step(self, model, inputs):
        assert self.args.per_device_train_batch_size == 64 and self.args.gradient_accumulation_steps == 1
        cuda = self.args.device.type == 'cuda'
        if self.profiled_steps < 3 and cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        loss = super().training_step(model, inputs)
        raw = self.accelerator.unwrap_model(model)
        if self.profiled_steps < 3 or abs(raw.step - raw.frozen_steps_for_regfrozen) <= 2:
            old.check_gradients(raw)
        if self.profiled_steps < 3:
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            record = dict(optimizer_step=self.state.global_step, local_batch=len(inputs['labels']),
                world_size=self.args.world_size, gradient_accumulation_steps=1,
                forward_backward_seconds=elapsed, examples_per_second=len(inputs['labels']) / elapsed,
                measurement='Forward/backward including input transfer; excludes dataloader wait and optimizer update.',
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                reserved_GiB=torch.cuda.memory_reserved() / 2**30 if cuda else None,
                finite_loss=bool(torch.isfinite(loss)))
            with (Path(self.args.output_dir) / 'batch64_profile.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        self.profiled_steps += 1
        return loss


def configuration(recipe, output):
    original = parent.read_json(output / 'SOURCE_RUN_CONFIG.json')
    origin = parent.read_json(output / 'IMPORT_COMPLETE.json')
    assert original['source_sha256'] == ORIGINAL_SOURCES, 'Original training sources changed; inspect before resuming'
    assert original['global_batch'] == 64 and original['steps_per_epoch'] == 446
    return dict(original, recipe='libri100_bpe500_local64__' + recipe, source_sha256=source_hashes(),
        world_size=1, nodes=1, local_batch=64, global_batch=64, gradient_accumulation_steps=1,
        initialization='Continue imported complete checkpoint; optimizer, scheduler and epoch/step unchanged.',
        resume_origin=origin, checkpoint_policy='Atomic epoch weights; dev-best3 plus last10, best-dev and latest2 full states.',
        batch_change='Explicit user correction: local64. One GPU chosen to retain global64 and reduce card allocation.',
        sampling_note=origin['sampling_note'])


def run(args):
    assert os.environ.get('PBS_JOBID') and torch.cuda.is_available()
    assert int(os.environ.get('WORLD_SIZE', 0)) == 1, 'One allocated GPU/process required'
    assert 180 <= args.max_seconds <= 10800
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.set_num_threads(4)
    base.set_seed(42)
    started = time.monotonic()
    output = HERE / 'models' / args.recipe
    config = configuration(args.recipe, output)
    submissions = parent.read_json(HERE / 'SUBMISSIONS.json') if (HERE / 'SUBMISSIONS.json').exists() else []
    for record in submissions:
        if record['job'].split('.')[0] == os.environ['PBS_JOBID'].split('.')[0]:
            assert record['recipe'] == args.recipe and record['source_sha256'] == config['source_sha256'], 'Submitted source contract changed'
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = output / 'run_config.json'
        if contract.exists():
            assert parent.read_json(contract) == config, 'Local64 resume contract changed'
        else:
            parent.write_json(contract, config)
            for relative in config['source_sha256']:
                target = output / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
        if (output / 'BENCHMARK_COMPLETE.json').exists():
            return 0
        _, rows = old.load_rows(False)
        tokenizer = old.prepare.Tokenizer(old.prepare.DATA / 'tokenizer')
        topology = base.build_topology(tokenizer, 'spottc_s3t2') if args.recipe.startswith('spottc') else None
        model = old.make_model(config)
        processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
            do_normalize=True, return_attention_mask=True)
        processor.save_pretrained(output / 'processor')
        if topology:
            parent.write_json(output / 'state_labels.json', topology['labels'])
            parent.write_json(output / 'token_to_states.json', topology['mapping'])
        validation = rows['dev-clean'] + rows['dev-other']
        dataset = {split: base.Dataset.from_list([dict(wav=row['wav'], labels=row['labels'],
            length=round(row['duration'] * 16000)) for row in items]) for split, items in [('train', rows['train']), ('dev', validation)]}
        arguments = TrainingArguments(output_dir=str(output), per_device_train_batch_size=64,
            per_device_eval_batch_size=2, gradient_accumulation_steps=1, num_train_epochs=50,
            learning_rate=2e-4, warmup_steps=500, weight_decay=.005, bf16=True, seed=42, data_seed=42,
            evaluation_strategy='epoch', save_strategy='epoch', save_total_limit=None, logging_steps=20,
            logging_nan_inf_filter=False, report_to=[], dataloader_num_workers=4, remove_unused_columns=False,
            group_by_length=True, length_column_name='length', max_grad_norm=1., disable_tqdm=True,
            save_safetensors=True, ddp_find_unused_parameters=True, ddp_broadcast_buffers=False)
        tracking = old.start_tracking(output, config)
        trainer = SingleTrainer(model=model, args=arguments, train_dataset=dataset['train'], eval_dataset=dataset['dev'],
            tokenizer=processor, data_collator=base.Collator(processor, topology['mapping'] if topology else None),
            compute_metrics=old.metrics_function(tokenizer, topology, len(rows['dev-clean'])),
            callbacks=[parent.Policy(output, config, started + args.max_seconds, tracking),
                       CheckpointRetentionCallback(output)])
        resume = parent.latest_checkpoint(output)
        assert resume is not None and parent.read_json(resume / 'trainer_state.json')['train_batch_size'] == 64
        assert old.prepare.sha256(resume / 'model.safetensors') == parent.read_json(resume / 'COMPLETE.json')['model_sha256']
        try:
            if (output / 'TRAIN_COMPLETE.json').exists():
                trainer._load_from_checkpoint(str(resume))
            else:
                trainer.train(resume_from_checkpoint=str(resume), ignore_keys_for_eval=['hidden_states', 'attentions'])
                checkpoint = parent.latest_checkpoint(output)
                assert checkpoint is not None
                complete = trainer.state.epoch >= 50 - 1e-5
                state = dict(epoch=trainer.state.epoch, step=trainer.state.global_step,
                    checkpoint=str(checkpoint), complete=complete, world_size=1, job=os.environ['PBS_JOBID'])
                parent.write_json(output / 'LATEST_STATE.json', state)
                if not complete:
                    return 75
                assert trainer.state.global_step == 22300
                parent.write_json(output / 'TRAIN_COMPLETE.json', state)
        finally:
            tracking.finish()
        from reproduction.libri100_bpe500_jsd_20261021.evaluate import benchmarks
        benchmarks(model, rows, processor, tokenizer, topology, output, config)
        return 0


def check():
    prepare.check()
    with tempfile.TemporaryDirectory() as directory:
        source, target = Path(directory) / 'source', Path(directory) / 'target'
        source.mkdir()
        data = base.Dataset.from_list([dict(input_values=[float(index % 7)] * 8, labels=[1]) for index in range(192)])
        config = dict(source_sha256={}, data=dict(tokenizer_sha256='fixture'))
        def trainer(output, stop=None):
            args = TrainingArguments(output_dir=str(output), use_cpu=True, per_device_train_batch_size=64,
                per_device_eval_batch_size=64, num_train_epochs=2, learning_rate=1e-4,
                save_strategy='epoch', evaluation_strategy='epoch', save_total_limit=None,
                report_to=[], remove_unused_columns=False, disable_tqdm=True)
            return SingleTrainer(model=old.ToyModel(PretrainedConfig()), args=args, train_dataset=data, eval_dataset=data,
                compute_metrics=lambda prediction: {'WER_percent': 90.},
                callbacks=[base.Policy(output, config, float('inf'), SimpleNamespace(log=lambda record: None), stop)])
        first = trainer(source, 3)
        first.train()
        checkpoint = parent.latest_checkpoint(source)
        state = parent.read_json(checkpoint / 'trainer_state.json')
        parent.write_json(checkpoint / 'trainer_state.json', dict(state, train_batch_size=32))
        for rank in [0, 1]:
            os.link(checkpoint / 'rng_state.pth', checkpoint / f'rng_state_{rank}.pth')
        marker = parent.read_json(checkpoint / 'COMPLETE.json')
        marker['files'].pop('rng_state.pth')
        for name in ['trainer_state.json', 'rng_state_0.pth', 'rng_state_1.pth']:
            marker['files'][name] = (checkpoint / name).stat().st_size
        parent.write_json(checkpoint / 'COMPLETE.json', dict(marker, world_size=2))
        parent.write_json(source / 'run_config.json', dict(local_batch=32, global_batch=64,
            world_size=2, epochs=50, planned_optimizer_steps=22300))
        prepare.snapshot(source, target)
        resumed = trainer(target)
        resumed.train(resume_from_checkpoint=str(parent.latest_checkpoint(target)))
        assert resumed.state.global_step == 6 and resumed.state.epoch == 2
        assert resumed._train_batch_size == 64
        assert (target / 'epoch_weights/epoch-002.pt').exists()
    for recipe in old.RECIPES:
        config = configuration(recipe, HERE / 'models' / recipe)
        assert config['local_batch'] == config['global_batch'] == 64 and config['world_size'] == 1
        assert config['epochs'] == 50 and config['planned_optimizer_steps'] == 22300
    print('PASS: checkpoint local32 -> local64 resumes correct epoch/step and unchanged global64 schedule')


def submit(recipe, dry_run=False):
    output = HERE / 'models' / recipe
    config = configuration(recipe, output)
    queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
    live = {line.split()[0].split('.')[0] for line in queue.splitlines()
            if line.split() and line.split()[0].split('.')[0].isdigit()}
    assert not live & {'3656067', '3656069'}, 'Old local32 jobs are still stopping; do not duplicate training'
    jobs = HERE / 'jobs'
    jobs.mkdir(exist_ok=True)
    path = HERE / 'SUBMISSIONS.json'
    records = parent.read_json(path) if path.exists() else []
    for record in records:
        if record['recipe'] == recipe and record['job'].split('.')[0] in live:
            print(json.dumps(dict(already_active=record['job'])))
            return
    log = jobs / (recipe + '_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log')
    command = [shutil.which('qsub') or 'qsub', '-N', 'l64_' + recipe, '-v', 'LIBRI_RECIPE=' + recipe,
               '-o', str(log), str(HERE / 'run.pbs')]
    if dry_run:
        print(json.dumps(dict(command=command, local_batch=64, global_batch=64, gpus=1,
            resume_epoch=config['resume_origin']['epoch'], resume_step=config['resume_origin']['step']), indent=2))
        return
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    job = result.stdout.strip()
    assert job.split('.')[0].isdigit(), job
    records.append(dict(job=job, recipe=recipe, log=str(log), command=command, local_batch=64, global_batch=64,
        world_size=1, source_sha256=config['source_sha256'], resume_origin=config['resume_origin'], evaluation_uploads=False))
    parent.write_json(path, records)
    print(json.dumps(dict(submitted=job, recipe=recipe, local_batch=64, gpus=1), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--submit', action='store_true')
    parser.add_argument('--recipe', choices=old.RECIPES, default='ottc_jsd')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.dry_run and not args.submit:
        parser.error('--dry-run requires --submit')
    if args.check:
        check()
    elif args.submit:
        (HERE / 'jobs').mkdir(exist_ok=True)
        with (HERE / 'jobs/.submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(args.recipe, args.dry_run)
    else:
        try:
            result = run(args)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        sys.exit(0 if result == 75 else result)