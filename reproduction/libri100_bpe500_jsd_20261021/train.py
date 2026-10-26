"""Two-GPU original OTTC/SPOTTC JSD on Libri100 with global batch64."""
import argparse
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.libri100_bpe500_jsd_20261021 import prepare
from reproduction.timit_bpe500_stages_20261020 import train as parent
from reproduction.powsm_bpe_timit_20261019.gradient_check_fix.train import check_gradients
from reproduction.checkpoint_retention import CheckpointRetentionCallback
from transformers import Trainer, TrainerCallback, TrainingArguments, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
import torch
import torch.distributed as dist

base = parent.base
base.normalize_text = prepare.normalize_text
RECIPES = ('ottc_jsd', 'spottc_jsd')


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def initialize(cpu=False):
    assert int(os.environ.get('WORLD_SIZE', 0)) == 2, 'Use torchrun with exactly two processes'
    torch.set_num_threads(2 if cpu else 4)
    if not cpu:
        assert os.environ.get('PBS_JOBID') and torch.cuda.is_available(), 'Allocated PBS GPUs required'
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('gloo' if cpu else 'nccl', timeout=timedelta(hours=2))


def source_hashes():
    sources = parent.source_hashes()
    paths = [*HERE.glob('*.py'), *HERE.glob('*.pbs'), *HERE.glob('*.sh'), ROOT / 'reproduction/checkpoint_retention.py',
             ROOT / 'reproduction/native_stride_ottc_20261018/graphs.py',
             ROOT / 'reproduction/native_stride_ottc_20261018/evaluate.py',
             ROOT / 'reproduction/timit_bpe500_50ep_20261020/segmentation.py',
             ROOT / 'reproduction/english_suite/mfa_protocol_comparison/scorer_source/boundary_rvalue.py']
    sources.update({str(path.relative_to(ROOT)): prepare.sha256(path) for path in paths})
    return sources


class DistributedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(**inputs)
        loss = output.loss / len(inputs['labels'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite original OTTC loss')
        return (loss, output) if return_outputs else loss

    def training_step(self, model, inputs):
        assert self.args.gradient_accumulation_steps == 1
        loss = super().training_step(model, inputs)
        raw = self.accelerator.unwrap_model(model)
        if self.state.global_step < 3 or abs(raw.step - raw.frozen_steps_for_regfrozen) <= 2:
            check_gradients(raw)
            if hasattr(raw, 'wav2vec2'):
                encoder_gradient = raw.wav2vec2.encoder.layers[0].attention.q_proj.weight.grad
                assert encoder_gradient is not None and torch.isfinite(encoder_gradient).all()
                assert bool(encoder_gradient.abs().sum() > 0), 'Missing encoder gradient'
                assert all(parameter.grad is None for parameter in raw.wav2vec2.feature_extractor.parameters())
            gradient = raw.lm_head.weight.grad.flatten()[:8].detach().contiguous()
            gathered = [torch.empty_like(gradient) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, gradient)
            for other in gathered:
                torch.testing.assert_close(other, gradient, rtol=1e-5, atol=1e-6)
            if primary():
                print(json.dumps(dict(step=self.state.global_step, local_batch=len(inputs['labels']),
                    world_size=2, gradient_accumulation_steps=1, synchronized_head_gradients=True,
                    phase='frozen' if raw.step >= raw.frozen_steps_for_regfrozen - 1 else 'joint')), flush=True)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, output = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        ids = output.logits.argmax(-1).masked_fill(output.hidden_states[1] == 0, 0)
        return loss.detach(), ids.detach(), inputs['labels'].detach()

    def _save_checkpoint(self, model, trial, metrics=None):
        destination = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
        if primary() and destination.exists() and not parent.valid_checkpoint(destination):
            destination.rename(destination.with_name(f'.incomplete-{destination.name}-{time.time_ns()}'))
        barrier()
        super()._save_checkpoint(model, trial, metrics)
        barrier()
        if primary():
            required = ['model.safetensors', 'config.json', 'trainer_state.json', 'optimizer.pt',
                        'scheduler.pt', 'training_args.bin', 'rng_state_0.pth', 'rng_state_1.pth']
            assert all((destination / name).is_file() for name in required)
            from safetensors import safe_open
            with safe_open(str(destination / 'model.safetensors'), framework='numpy') as weights:
                assert weights.keys()
            for name in ['optimizer.pt', 'scheduler.pt', 'rng_state_0.pth', 'rng_state_1.pth']:
                with zipfile.ZipFile(destination / name) as archive:
                    assert archive.namelist()
            parent.write_json(destination / 'COMPLETE.json', dict(step=self.state.global_step, world_size=2,
                files={name: (destination / name).stat().st_size for name in required},
                model_sha256=prepare.sha256(destination / 'model.safetensors')))
            completed = sorted((path for path in destination.parent.glob('checkpoint-[0-9]*')
                if parent.valid_checkpoint(path)), key=lambda path: int(path.name.split('-')[1]))
            for obsolete in completed[:-2]:
                shutil.rmtree(obsolete)
        barrier()


class Policy(TrainerCallback):
    def __init__(self, output, config, tracking, deadline, stop=None):
        self.output, self.config, self.tracking, self.deadline, self.stop = output, config, tracking, deadline, stop
        path = output / 'BEST_DEV.json'
        self.best = parent.read_json(path)['WER_percent'] if path.exists() else float('inf')

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        model.step = state.global_step - 1

    def on_step_end(self, args, state, control, **kwargs):
        stop = primary() and (time.monotonic() >= self.deadline or (self.stop is not None and state.global_step >= self.stop))
        flag = torch.tensor(int(bool(stop)), device=args.device)
        dist.broadcast(flag, src=0)
        if flag.item():
            control.should_save = True
            control.should_training_stop = True
        elif state.global_step % 200 == 0:
            control.should_save = True
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if not primary() or abs(state.epoch - epoch) > 1e-6:
            return
        path = self.output / 'epoch_weights' / f'epoch-{epoch:03d}.pt'
        metadata = dict(epoch=epoch, step=state.global_step, model_config=model.config.to_dict(),
                        source_sha256=self.config['source_sha256'], tokenizer_sha256=self.config['tokenizer_sha256'])
        if not path.exists():
            base.atomic_weights(model, path, metadata)
            parent.write_json(path.with_suffix('.json'), dict(**metadata, weights_sha256=prepare.sha256(path)))

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not primary():
            return
        value = metrics.get('eval_WER_percent')
        if value is not None and value < self.best:
            self.best = value
            path = self.output / 'best_dev.pt'
            base.atomic_weights(model, path, dict(epoch=state.epoch, step=state.global_step))
            parent.write_json(self.output / 'BEST_DEV.json', dict(epoch=state.epoch, step=state.global_step,
                WER_percent=value, checkpoint=str(path), weights_sha256=prepare.sha256(path),
                selection='Pooled full dev-clean + dev-other greedy WER; no test selection'))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if primary():
            record = dict(logs or {}, epoch=state.epoch, optimizer_step=state.global_step)
            with (self.output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            self.tracking.log(record)


class DistributedRetention(CheckpointRetentionCallback):
    def __init__(self, output):
        if primary():
            super().__init__(output)

    def on_save(self, args, state, control, **kwargs):
        if not primary():
            return control
        checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        alias = checkpoint / 'rng_state.pth'
        if not alias.exists():
            os.link(checkpoint / 'rng_state_0.pth', alias)
        return super().on_save(args, state, control, **kwargs)


def load_rows(smoke=False):
    metadata = parent.read_json(prepare.DATA / 'metadata.json')
    rows = {}
    for split in prepare.SPLITS:
        path = prepare.DATA / (split + '.jsonl')
        assert prepare.sha256(path) == metadata['manifests_sha256'][split]
        rows[split] = [json.loads(line) for line in path.open()]
    if smoke:
        rows['train'] = sorted(rows['train'], key=lambda row: row['duration'], reverse=True)[:192]
        for split in prepare.SPLITS[1:]:
            rows[split] = rows[split][:4]
    return metadata, rows


def configuration(recipe, metadata, rows, smoke):
    steps = math.ceil(math.ceil(len(rows['train']) / 2) / 32)
    return dict(recipe=HERE.name + '__' + recipe, stage=recipe, phase='smoke' if smoke else 'full',
        data=metadata, tokenizer_sha256=metadata['tokenizer_sha256'], source_sha256=source_hashes(),
        epochs=1 if smoke else 50, world_size=2, nodes=2, gpus_per_node=1,
        local_batch=32, global_batch=64, gradient_accumulation_steps=1,
        steps_per_epoch=steps, planned_optimizer_steps=steps if smoke else 50 * steps,
        seed=42, learning_rate=2e-4, warmup_steps=0 if smoke else 500, weight_decay=.005,
        backbone_sha256=prepare.sha256(parent.BACKBONE / 'pytorch_model.bin'),
        output_classes=1498 if recipe.startswith('spottc') else 500, bins_metric='jsd', forward_type='regfrozen',
        frozen_steps_for_regfrozen=2 if smoke else 30 * steps + 1,
        topology='s3_t2 state-uniform targets' if recipe.startswith('spottc') else 'none',
        decoding=['greedy', 's3_t2_star_star constrained S0/S2'] if recipe.startswith('spottc') else ['greedy'],
        initialization='Independent official Wav2Vec2-large-LV60 SSL; random prediction/mass heads; no trained TIMIT or phone weights.',
        objective='Original upstream OTTC JSD; no replacement divergence, no topology/FSA training loss.',
        loss_reduction='Original summed OTTC loss / actual local batch, averaged across two DDP ranks.',
        activation_checkpointing='Non-reentrant Wav2Vec2 checkpointing, no microbatch splitting.',
        selection='Pooled dev-clean/dev-other WER; clean/other and decoding modes reported separately.',
        checkpoint_policy='Every epoch atomic weights; best3 plus last10 and dependencies; best-dev; latest2 complete full states with both rank RNGs.',
        evaluation_uploads=False, smoke=smoke,
        smoke_utterances=[row['utt'] for row in rows['train']] if smoke else None,
        segmentation='Separate zero-shot TIMIT human word-boundary evaluation; Libri automatic phone alignments are not human ground truth.')


def make_model(config):
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
    model = wvForOTTC(wvOTTCConfig(wav2vec2_path=str(parent.BACKBONE), vocab_size=config['output_classes'],
        hidden_size=1024, bins_metric='jsd', forward_type='regfrozen', pad_token_id=0,
        frozen_steps_for_regfrozen=config['frozen_steps_for_regfrozen']))
    model.freeze_feature_encoder()
    model.wav2vec2_freeze.requires_grad_(False)
    model.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.wav2vec2_freeze.gradient_checkpointing_disable()
    assert model.wav2vec2.encoder._gradient_checkpointing_func.keywords['use_reentrant'] is False
    return model


def metrics_function(tokenizer, topology, clean_count):
    decode_metrics = base.compute_metrics(tokenizer, topology)
    def metrics(prediction):
        values = decode_metrics(prediction)
        for split, selection in [('clean', slice(0, clean_count)), ('other', slice(clean_count, None))]:
            subset = SimpleNamespace(label_ids=prediction.label_ids[selection], predictions=prediction.predictions[selection])
            scores = decode_metrics(subset)
            values[split + '_WER_percent'] = scores['WER_percent']
            values[split + '_CER_percent'] = scores['CER_percent']
        return values
    return metrics


def start_tracking(output, config):
    if not primary():
        return SimpleNamespace(log=lambda record: None, finish=lambda: None)
    if config['smoke']:
        return base.init_evaluation(project='ottc-timit-reproduction', name=config['recipe'] + '_smoke',
                                    config=config, dir=str(output))
    import wandb
    defaults = parent.read_json(ROOT / 'reproduction/wandb_config.json')
    path = output / 'wandb_run.json'
    saved = parent.read_json(path) if path.exists() else {}
    options = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
        entity=os.environ.get('WANDB_ENTITY', defaults['entity']), id=saved.get('id') or wandb.util.generate_id(),
        name=config['recipe'], group=HERE.name, job_type='training', config=config, dir=str(output),
        mode=os.environ.get('WANDB_MODE', defaults['mode']), settings=wandb.Settings(init_timeout=45, disable_code=True))
    parent.write_json(path, {key: options[key] for key in ['id', 'project', 'entity']})
    if saved and options['mode'] == 'online':
        options['resume'] = 'allow'
    try:
        return wandb.init(**options)
    except wandb.errors.CommError:
        options['mode'] = 'offline'
        options.pop('resume', None)
        return wandb.init(**options)


def run(args):
    initialize()
    assert 180 <= args.max_seconds <= 10800
    assert args.stop_after_steps is None or (args.smoke and args.stop_after_steps == 1)
    started = time.monotonic()
    os.environ['OTTC_EVAL_UPLOAD'] = '0'
    base.set_seed(42)
    metadata, rows = load_rows(args.smoke)
    config = configuration(args.recipe, metadata, rows, args.smoke)
    output = HERE / ('smoke' if args.smoke else 'models') / args.recipe
    lock = None
    if primary():
        output.mkdir(parents=True, exist_ok=True)
        lock = (output / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submissions = parent.read_json(HERE / 'SUBMISSIONS.json') if (HERE / 'SUBMISSIONS.json').exists() else []
        for record in submissions:
            if record['job'].split('.')[0] == os.environ['PBS_JOBID'].split('.')[0]:
                assert record['recipe'] == args.recipe and record['source_sha256'] == config['source_sha256']
        contract = output / 'run_config.json'
        if contract.exists():
            assert parent.read_json(contract) == config, 'Existing run/resume contract differs'
        else:
            parent.write_json(contract, config)
            for relative in config['source_sha256']:
                target = output / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
    barrier()
    try:
        completion = output / ('PASS.json' if args.smoke else 'BENCHMARK_COMPLETE.json')
        if completion.exists():
            return 0
        if not args.smoke:
            gate = parent.read_json(HERE / 'smoke' / args.recipe / 'PASS.json')
            assert gate['source_sha256'] == config['source_sha256'] and gate['tokenizer_sha256'] == metadata['tokenizer_sha256']
            assert gate['world_size'] == 2 and gate['local_batch'] == 32
        tokenizer = prepare.Tokenizer(prepare.DATA / 'tokenizer')
        topology = base.build_topology(tokenizer, 'spottc_s3t2') if args.recipe.startswith('spottc') else None
        model = make_model(config)
        processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
            do_normalize=True, return_attention_mask=True)
        if primary():
            processor.save_pretrained(output / 'processor')
            if topology:
                parent.write_json(output / 'state_labels.json', topology['labels'])
                parent.write_json(output / 'token_to_states.json', topology['mapping'])
        validation = rows['dev-clean'] + rows['dev-other']
        dataset = {split: base.Dataset.from_list([dict(wav=row['wav'], labels=row['labels'],
            length=round(row['duration'] * 16000)) for row in items]) for split, items in [('train', rows['train']), ('dev', validation)]}
        arguments = TrainingArguments(output_dir=str(output), per_device_train_batch_size=32,
            per_device_eval_batch_size=2, gradient_accumulation_steps=1, num_train_epochs=config['epochs'],
            learning_rate=2e-4, warmup_steps=config['warmup_steps'], weight_decay=.005, bf16=True,
            seed=42, data_seed=42, evaluation_strategy='epoch', save_strategy='epoch', save_total_limit=None,
            logging_steps=1 if args.smoke else 20, logging_nan_inf_filter=False, report_to=[],
            dataloader_num_workers=2, remove_unused_columns=False, group_by_length=True, length_column_name='length',
            max_grad_norm=1., disable_tqdm=True, save_safetensors=True, ddp_find_unused_parameters=True,
            ddp_broadcast_buffers=False, ddp_timeout=7200)
        tracking = start_tracking(output, config)
        trainer = DistributedTrainer(model=model, args=arguments, train_dataset=dataset['train'],
            eval_dataset=dataset['dev'], tokenizer=processor,
            data_collator=base.Collator(processor, topology['mapping'] if topology else None),
            compute_metrics=metrics_function(tokenizer, topology, len(rows['dev-clean'])),
            callbacks=[Policy(output, config, tracking, started + args.max_seconds, args.stop_after_steps),
                       DistributedRetention(output)])
        resume = parent.latest_checkpoint(output)
        if resume:
            if primary():
                assert prepare.sha256(resume / 'model.safetensors') == parent.read_json(resume / 'COMPLETE.json')['model_sha256']
                parent.write_json(output / 'RESUME_ORIGIN.json', dict(checkpoint=str(resume),
                    step=parent.read_json(resume / 'trainer_state.json')['global_step']))
            barrier()
        try:
            if (output / 'TRAIN_COMPLETE.json').exists():
                assert resume is not None
                trainer._load_from_checkpoint(str(resume))
            else:
                trainer.train(resume_from_checkpoint=str(resume) if resume else None,
                              ignore_keys_for_eval=['hidden_states', 'attentions'])
                checkpoint = parent.latest_checkpoint(output)
                assert checkpoint is not None
                complete = trainer.state.epoch >= config['epochs'] - 1e-5
                if primary():
                    state = dict(epoch=trainer.state.epoch, step=trainer.state.global_step,
                        checkpoint=str(checkpoint), complete=complete, world_size=2, job=os.environ['PBS_JOBID'])
                    parent.write_json(output / 'LATEST_STATE.json', state)
                    if complete:
                        assert trainer.state.global_step == config['planned_optimizer_steps']
                        parent.write_json(output / 'TRAIN_COMPLETE.json', state)
                barrier()
                if not complete:
                    return 75
        finally:
            tracking.finish()
        if args.smoke:
            assert (output / 'RESUME_ORIGIN.json').exists()
            memory = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=arguments.device)
            gathered = [torch.zeros_like(memory) for _ in range(2)]
            dist.all_gather(gathered, memory)
            if primary():
                parent.write_json(output / 'PASS.json', dict(recipe=args.recipe, steps=3, world_size=2, local_batch=32,
                    output_classes=config['output_classes'], source_sha256=config['source_sha256'],
                    tokenizer_sha256=metadata['tokenizer_sha256'], peak_cuda_GiB=[float(item.item()) for item in gathered],
                    actual_resume=True, phases=['joint', 'frozen'], evaluation_uploads=False))
            barrier()
            return 0
        from reproduction.libri100_bpe500_jsd_20261021.evaluate import benchmarks
        benchmarks(model, rows, processor, tokenizer, topology, output, config)
        return 0
    finally:
        if lock is not None:
            lock.close()


def submit(recipe, dry_run=False):
    assert recipe in RECIPES
    metadata, rows = load_rows(False)
    config = configuration(recipe, metadata, rows, False)
    jobs = HERE / 'jobs'
    jobs.mkdir(exist_ok=True)
    log = jobs / (recipe + '_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log')
    command = [shutil.which('qsub') or 'qsub', '-N', 'l100_' + recipe, '-v', 'LIBRI_RECIPE=' + recipe,
               '-o', str(log), str(HERE / 'run.pbs')]
    if dry_run:
        print(json.dumps(dict(command=command, epochs=50, world_size=2, local_batch=32, global_batch=64,
            train_utterances=len(rows['train']), steps=config['planned_optimizer_steps'], walltime='04:00:00',
            independent_queue=True, evaluation_uploads=False), indent=2))
        return
    with (jobs / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = HERE / 'SUBMISSIONS.json'
        records = parent.read_json(path) if path.exists() else []
        queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
        live = {line.split()[0].split('.')[0] for line in queue.splitlines()
                if line.split() and line.split()[0].split('.')[0].isdigit()}
        for record in records:
            if record['recipe'] == recipe and record['job'].split('.')[0] in live:
                assert record['source_sha256'] == config['source_sha256']
                print(json.dumps(dict(already_active=record['job'], recipe=recipe)))
                return
        output = HERE / 'models' / recipe
        if (output / 'BENCHMARK_COMPLETE.json').exists():
            assert parent.read_json(output / 'run_config.json') == config
            print(json.dumps(dict(completed=True, recipe=recipe)))
            return
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'qsub failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}')
        job = result.stdout.strip()
        assert job.split('.')[0].isdigit(), job
        records.append(dict(job=job, recipe=recipe, command=command, log=str(log), source_sha256=config['source_sha256'],
            data_metadata_sha256=prepare.sha256(prepare.DATA / 'metadata.json'), epochs=50, world_size=2,
            global_batch=64, local_batch=32, walltime='04:00:00', evaluation_uploads=False))
        parent.write_json(path, records)
        print(json.dumps(dict(submitted=job, recipe=recipe, log=str(log)), indent=2), flush=True)


class ToyModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.shared = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 4)
        self.lm_weight = torch.nn.Linear(8, 1)
        self.step, self.frozen_steps_for_regfrozen = -1, 2

    def forward(self, input_values, labels=None, attention_mask=None):
        from torch.utils.checkpoint import checkpoint
        hidden = checkpoint(self.shared, input_values, use_reentrant=False)
        logits = self.lm_head(hidden)
        alignment = self.lm_weight(hidden)
        if self.step >= self.frozen_steps_for_regfrozen - 1:
            alignment = alignment.detach()
        loss = logits.square().sum() + alignment.square().sum()
        return CausalLMOutput(loss=loss, logits=logits[:, None, :],
            hidden_states=(None, torch.ones(len(logits), 1, dtype=torch.long, device=logits.device)))


def check_ddp():
    initialize(cpu=True)
    directory = [tempfile.mkdtemp(prefix='libri-ddp-', dir=HERE) if primary() else None]
    dist.broadcast_object_list(directory, src=0)
    output = Path(directory[0])
    dataset = base.Dataset.from_list([dict(input_values=[float(index % 7)] * 8, labels=[1]) for index in range(192)])
    config = dict(source_sha256={}, tokenizer_sha256='CPU fixture')
    def trainer(stop=None):
        base.set_seed(42)
        args = TrainingArguments(output_dir=str(output), use_cpu=True, per_device_train_batch_size=32,
            num_train_epochs=1, learning_rate=1e-4, gradient_accumulation_steps=1,
            save_strategy='epoch', save_total_limit=None, report_to=[], remove_unused_columns=False,
            disable_tqdm=True, ddp_find_unused_parameters=True, ddp_broadcast_buffers=False)
        return DistributedTrainer(model=ToyModel(PretrainedConfig()), args=args, train_dataset=dataset,
            callbacks=[Policy(output, config, SimpleNamespace(log=lambda value: None), float('inf'), stop)])
    first = trainer(1)
    first.train()
    assert first.state.global_step == 1
    resumed = trainer()
    resumed.train(resume_from_checkpoint=str(parent.latest_checkpoint(output)))
    assert resumed.state.global_step == 3 and resumed.state.epoch == 1
    barrier()
    if primary():
        checkpoint = parent.latest_checkpoint(output)
        assert parent.read_json(checkpoint / 'COMPLETE.json')['world_size'] == 2
        assert all((checkpoint / f'rng_state_{rank}.pth').exists() for rank in range(2))
        assert (output / 'epoch_weights/epoch-001.pt').exists()
        print('PASS: two-rank gradients, non-reentrant checkpointing, frozen head, optimizer/scheduler/RNG resume and atomic epoch archive', flush=True)
    barrier()
    if primary():
        shutil.rmtree(output)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-ddp', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--submit', action='store_true')
    parser.add_argument('--recipe', choices=RECIPES, default='ottc_jsd')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--max-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.dry_run and not args.submit:
        parser.error('--dry-run requires --submit')
    if args.check_ddp:
        check_ddp()
    elif args.check:
        prepare.check()
        from reproduction.libri100_bpe500_jsd_20261021.decoder import check
        check()
        metadata, rows = load_rows(False)
        for recipe in RECIPES:
            config = configuration(recipe, metadata, rows, False)
            assert config['planned_optimizer_steps'] == 22300 and config['bins_metric'] == 'jsd'
            assert config['world_size'] * config['local_batch'] == config['global_batch'] == 64
        print('PASS: both recipes use 50 epochs/22300 steps, original JSD and global64/local32 DDP')
    elif args.submit:
        submit(args.recipe, args.dry_run)
    else:
        try:
            result = run(args)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        if result == 75:
            if int(os.environ.get('RANK', 0)) == 0:
                print('PAUSED: complete checkpoint saved; PBS wrapper inspects completion markers.', flush=True)
            result = 0
        sys.exit(result)


if __name__ == '__main__':
    main()