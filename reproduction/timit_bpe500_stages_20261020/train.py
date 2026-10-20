"""Independent BPE500 CTC -> OTTC CE -> SPOTTC CE -> SPOTTC JSD stages."""
import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.timit_bpe500_20261020 import train as bpe
from reproduction.powsm_bpe_timit_20261019.ctc.output_fix import train as ctc
from reproduction.powsm_bpe_timit_20261019.ctc.resume_fix.train import Wav2Vec2Config
from accelerate.utils import convert_to_fp32

base = bpe.base
torch = base.torch
base.normalize_text = bpe.normalize_text
STAGES = ('ctc', 'ottc_ce', 'spottc_ce', 'spottc_jsd')
DATA = bpe.HERE / 'data'
BACKBONE = ROOT / 'reproduction/models/wav2vec2-large-lv60'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_hashes():
    sources = dict(bpe.source_hashes())
    sources.update(ctc.source_hashes())
    paths = [Path(sys.modules[Wav2Vec2Config.__module__].__file__),
             ROOT / 'reproduction/wandb_config.json', *HERE.glob('*.py'), *HERE.glob('*.pbs')]
    sources.update({str(path.relative_to(ROOT)): base.sha256(path) for path in paths})
    return sources


def load_data():
    metadata = read_json(DATA / 'metadata.json')
    tokenizer = bpe.Tokenizer(DATA / 'tokenizer')
    for filename, key in [('bpe.model', 'tokenizer_sha256'), ('tokens.json', 'tokens_sha256')]:
        assert base.sha256(DATA / 'tokenizer' / filename) == metadata[key]
    rows = {}
    for split, expected in [('train', 4620), ('dev', 400), ('test', 192)]:
        path = DATA / (split + '.jsonl')
        assert base.sha256(path) == metadata['manifests_sha256'][split]
        rows[split] = [json.loads(line) for line in path.open()]
        assert len(rows[split]) == expected
        for row in rows[split]:
            assert row['labels'] and all(4 <= label < 500 for label in row['labels'])
            assert tokenizer.decode(row['labels']) == row['text'] == bpe.normalize_text(row['text'])
    return metadata, tokenizer, rows


def topology_for(stage, tokenizer):
    assert stage in STAGES
    return base.build_topology(tokenizer, 'spottc_s3t2') if stage.startswith('spottc') else None


def valid_checkpoint(path):
    marker = path / 'COMPLETE.json'
    if not marker.is_file():
        return False
    record = read_json(marker)
    return all((path / name).is_file() and (path / name).stat().st_size == size
               for name, size in record['files'].items())


def latest_checkpoint(output):
    candidates = sorted(output.glob('checkpoint-[0-9]*'), key=lambda path: int(path.name.split('-')[1]))
    return next((path for path in reversed(candidates) if valid_checkpoint(path)), None)


class CompleteCheckpoint:
    def _save_checkpoint(self, model, trial, metrics=None):
        destination = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
        if destination.exists() and not valid_checkpoint(destination):
            destination.rename(destination.with_name(f'.incomplete-{destination.name}-{time.time_ns()}'))
        super()._save_checkpoint(model, trial, metrics)
        required = ['model.safetensors', 'config.json', 'trainer_state.json', 'optimizer.pt',
                    'scheduler.pt', 'rng_state.pth', 'training_args.bin']
        assert all((destination / filename).is_file() for filename in required)
        from safetensors import safe_open
        with safe_open(str(destination / 'model.safetensors'), framework='numpy') as weights:
            assert weights.keys()
        assert read_json(destination / 'trainer_state.json')['global_step'] == self.state.global_step
        write_json(destination / 'COMPLETE.json', dict(step=self.state.global_step,
            files={filename: (destination / filename).stat().st_size for filename in required},
            model_sha256=base.sha256(destination / 'model.safetensors')))
        completed = sorted((path for path in Path(self.args.output_dir).glob('checkpoint-[0-9]*')
                            if valid_checkpoint(path)), key=lambda path: int(path.name.split('-')[1]))
        for obsolete in completed[:-2]:
            shutil.rmtree(obsolete)


class CTCTrainer(CompleteCheckpoint, ctc.original.CTCTrainer):
    pass


class OTTCTrainer(CompleteCheckpoint, bpe.checked.CheckedTrainer):
    pass


def fit_passed(metrics):
    return metrics['CER_percent'] <= 5 and metrics['WER_percent'] <= 10


def learning_passed(records, baseline):
    eligible = [row for row in records if row['step'] >= 500]
    return len(eligible) >= 3 and all(
        row['WER_percent'] < min(100., baseline['WER_percent']) and
        row['CER_percent'] < baseline['CER_percent'] for row in eligible[-3:])


def gate_path(stage, phase):
    return HERE / phase / stage / ('LEARNING_PASS.json' if phase == 'full' else 'PASS.json')


def require_gate(stage, phase, sources, metadata):
    path = gate_path(stage, phase)
    assert path.is_file(), f'Gate not passed: {path}'
    gate = read_json(path)
    assert gate['stage'] == stage and gate['phase'] == phase and gate['passed'] is True
    assert gate['source_sha256'] == sources and gate['data'] == metadata, f'Stale gate: {path}'
    return gate


def start_tracking(output, config):
    if config['phase'] != 'full':
        return base.init_evaluation(project='ottc-timit-reproduction', name=config['recipe'],
                                    config=config, dir=str(output))
    import wandb
    defaults = read_json(ROOT / 'reproduction/wandb_config.json')
    saved_path = output / 'wandb_run.json'
    saved = read_json(saved_path) if saved_path.exists() else {}
    options = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
        entity=os.environ.get('WANDB_ENTITY', defaults['entity']),
        id=saved.get('id') or wandb.util.generate_id(), name=config['recipe'],
        group=HERE.name, job_type='training', config=config, dir=str(output),
        mode=os.environ.get('WANDB_MODE', defaults['mode']),
        settings=wandb.Settings(init_timeout=45, disable_code=True))
    write_json(saved_path, {key: options[key] for key in ['id', 'project', 'entity']})
    if saved and options['mode'] == 'online':
        options['resume'] = 'allow'
    try:
        return wandb.init(**options)
    except wandb.errors.CommError:
        options['mode'] = 'offline'
        options.pop('resume', None)
        return wandb.init(**options)


class Policy(base.Policy):
    def __init__(self, *args, evaluation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation = evaluation

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, model=model, metrics=metrics, **kwargs)
        if self.config['phase'] == 'overfit' and fit_passed({
                'WER_percent': metrics['eval_WER_percent'], 'CER_percent': metrics['eval_CER_percent']}):
            control.should_training_stop = True
            control.should_save = True
        if self.config['phase'] == 'full' and self.evaluation is not None:
            rows, processor, tokenizer, topology = self.evaluation
            values = base.evaluate(model, rows, processor, tokenizer,
                self.output / 'development' / f'step-{state.global_step:06d}.jsonl', 2, topology)
            record = dict(values, step=state.global_step, epoch=state.epoch)
            path = self.output / 'development.jsonl'
            previous = [json.loads(line) for line in path.open()] if path.exists() else []
            previous = [row for row in previous if row['step'] < state.global_step]
            with path.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if learning_passed(previous + [record], read_json(self.output / 'INITIAL_DEV.json')):
                write_json(self.output / 'LEARNING_PASS.json', dict(stage=self.config['stage'], phase='full',
                    passed=True, step=state.global_step, data=self.config['data'],
                    source_sha256=self.config['source_sha256'], evidence=(previous + [record])[-3:],
                    protocol='Three post-warmup dev evaluations beat initialization in WER/CER; WER below 100.',
                    evaluation_uploads=False))
        return control


def run_training(args):
    assert torch.cuda.is_available() and os.environ.get('PBS_JOBID'), 'Allocated PBS GPU required'
    assert args.max_seconds >= 180
    assert args.stop_after_steps is None or (args.phase == 'smoke' and args.stop_after_steps == 1)
    os.environ['OTTC_EVAL_UPLOAD'] = '0'
    torch.set_num_threads(4)
    base.set_seed(42)
    started = time.monotonic()
    metadata, tokenizer, rows = load_data()
    sources = source_hashes()
    predecessor = STAGES.index(args.stage) - 1
    if predecessor >= 0:
        require_gate(STAGES[predecessor], 'full', sources, metadata)
    if args.phase in ('overfit', 'full'):
        require_gate(args.stage, 'smoke', sources, metadata)
    if args.phase == 'full':
        require_gate(args.stage, 'overfit', sources, metadata)
    diagnostic = args.phase != 'full'
    smoke = args.phase == 'smoke'
    if smoke:
        rows['train'] = sorted(rows['train'], key=lambda row: row['duration'], reverse=True)[:12]
        rows['dev'] = rows['dev'][:4]
    elif args.phase == 'overfit':
        indices = base.np.random.default_rng(42).choice(len(rows['train']), 32, replace=False)
        rows['train'] = [rows['train'][int(index)] for index in indices]
        rows['dev'] = rows['train']
    topology = topology_for(args.stage, tokenizer)
    classes = 500 if topology is None else len(topology['labels'])
    config = dict(name=HERE.name, recipe=HERE.name + '__' + args.stage + '__' + args.phase,
        stage=args.stage, phase=args.phase, source_sha256=sources, data=metadata, seed=42,
        epochs=1 if smoke else 40, global_batch=4 if diagnostic else 64, microbatch=2,
        learning_rate=2e-4, warmup_steps=0 if diagnostic else 500, weight_decay=.005, smoke=smoke,
        output_classes=classes, topology='s3_t2' if topology else 'none',
        topology_adaptation='All 499 nonblank IDs have three ordered states, including unused reserved IDs.' if topology else 'none',
        objective='ctc' if args.stage == 'ctc' else ('jsd' if args.stage.endswith('jsd') else 'crossentropy'),
        backbone_sha256=base.sha256(BACKBONE / 'pytorch_model.bin'),
        backbone_config_sha256=base.sha256(BACKBONE / 'config.json'),
        initialization='Independent official SSL backbone and random head; no cross-stage or diagnostic weight transfer.',
        train_utterances=[row['utt'] for row in rows['train']] if diagnostic else 'full 4620',
        spec_augment=args.phase != 'overfit', decoding='audio-only greedy; no LM; original state-reset decoder for S3T2',
        freeze_alignment_after_epochs=30, evaluation_uploads=False,
        loss_reduction='CTC target-length mean' if args.stage == 'ctc' else 'Original OTTC sum / logical batch',
        checkpoint_policy='Atomic weights every completed epoch; retain all; best-dev; latest two complete full states; incomplete states preserved.',
        diagnostic_protocol='Overfit: fixed 32 train utterances, zero warmup, SpecAugment off; WER<=10 and CER<=5; no test inference.')
    output = HERE / args.phase / args.stage
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = output / 'run_config.json'
        if contract.exists():
            assert read_json(contract) == config, 'Source, model or schedule contract changed'
        else:
            write_json(contract, config)
            for relative in sources:
                target = output / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
        completion = output / ('BENCHMARK_COMPLETE.json' if args.phase == 'full' else 'PASS.json')
        if completion.exists():
            require_gate(args.stage, args.phase, sources, metadata)
            return 0
        if (output / 'TRAIN_COMPLETE.json').exists() and diagnostic:
            raise RuntimeError('Completed diagnostic did not pass; inspect results before changing its recipe')
        if args.stage == 'ctc':
            model_config = Wav2Vec2Config.from_pretrained(str(BACKBONE), local_files_only=True,
                vocab_size=500, pad_token_id=0, ctc_loss_reduction='mean', ctc_zero_infinity=False,
                final_dropout=0.1, apply_spec_augment=config['spec_augment'])
            model = ctc.CTCModel.from_pretrained(str(BACKBONE), config=model_config, local_files_only=True)
            trainer_class = CTCTrainer
        else:
            from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
            steps_per_epoch = math.ceil(len(rows['train']) / config['global_batch'])
            model = wvForOTTC(wvOTTCConfig(wav2vec2_path=str(BACKBONE), vocab_size=classes,
                hidden_size=1024, bins_metric=config['objective'], forward_type='regfrozen',
                frozen_steps_for_regfrozen=2 if smoke else 30 * steps_per_epoch + 1, pad_token_id=0))
            model.wav2vec2_freeze.requires_grad_(False)
            model.wav2vec2.config.apply_spec_augment = config['spec_augment']
            model.wav2vec2_freeze.config.apply_spec_augment = config['spec_augment']
            trainer_class = OTTCTrainer
        model.freeze_feature_encoder()
        processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
            do_normalize=True, return_attention_mask=True)
        processor.save_pretrained(output / 'processor')
        if topology is not None:
            write_json(output / 'token_to_states.json', topology['mapping'])
            write_json(output / 'state_labels.json', topology['labels'])
        resume = latest_checkpoint(output)
        if resume:
            assert base.sha256(resume / 'model.safetensors') == read_json(resume / 'COMPLETE.json')['model_sha256']
            write_json(output / 'RESUME_ORIGIN.json', dict(checkpoint=str(resume),
                step=read_json(resume / 'trainer_state.json')['global_step']))
        if args.phase == 'full' and not (output / 'INITIAL_DEV.json').exists():
            assert resume is None, 'Initialization metrics missing from existing training'
            model.cuda()
            metrics = base.evaluate(model, rows['dev'], processor, tokenizer, output / 'initial/dev.jsonl', 2, topology)
            write_json(output / 'INITIAL_DEV.json', metrics)
        dataset = {split: base.Dataset.from_list([dict(wav=row['wav'], length=round(row['duration'] * 16000),
            labels=row['labels']) for row in rows[split]]) for split in ['train', 'dev']}
        run = start_tracking(output, config)
        training_args = base.TrainingArguments(output_dir=str(output),
            per_device_train_batch_size=config['global_batch'], per_device_eval_batch_size=2,
            num_train_epochs=config['epochs'], learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'],
            weight_decay=.005, bf16=True, seed=42, data_seed=42, evaluation_strategy='epoch',
            save_strategy='epoch', save_total_limit=None, logging_steps=1 if diagnostic else 10,
            logging_nan_inf_filter=False, report_to=[], dataloader_num_workers=2,
            remove_unused_columns=False, group_by_length=True, length_column_name='length',
            max_grad_norm=1., disable_tqdm=True, save_safetensors=True)
        trainer = trainer_class(model=model, args=training_args, train_dataset=dataset['train'],
            eval_dataset=dataset['dev'], tokenizer=processor,
            data_collator=base.ChunkedCollator(base.Collator(processor, topology['mapping'] if topology else None), 2),
            compute_metrics=base.compute_metrics(tokenizer, topology),
            callbacks=[Policy(output, config, started + args.max_seconds, run, args.stop_after_steps,
                              evaluation=(rows['dev'], processor, tokenizer, topology))])
        trainer.smoke_checks = smoke
        try:
            if (output / 'TRAIN_COMPLETE.json').exists():
                assert resume is not None
                trainer._load_from_checkpoint(str(resume))
                model.step = read_json(resume / 'trainer_state.json')['global_step'] - 1
            else:
                trainer.train(resume_from_checkpoint=str(resume) if resume else None,
                              ignore_keys_for_eval=['hidden_states', 'attentions'])
                checkpoint = latest_checkpoint(output)
                assert checkpoint is not None
                complete = trainer.state.epoch >= config['epochs'] - 1e-5
                write_json(output / 'LATEST_STATE.json', dict(epoch=trainer.state.epoch, step=trainer.state.global_step,
                    checkpoint=str(checkpoint), complete=complete, job=os.environ.get('PBS_JOBID')))
                if complete:
                    write_json(output / 'TRAIN_COMPLETE.json', dict(epoch=trainer.state.epoch, step=trainer.state.global_step))
                if diagnostic:
                    metrics = base.evaluate(model, rows['dev'], processor, tokenizer,
                        output / 'decode' / f'step-{trainer.state.global_step:06d}.jsonl', 2, topology)
                    passed = complete if smoke else fit_passed(metrics)
                    if smoke and passed:
                        assert resume is not None and trainer.state.global_step == 3
                        assert (output / 'epoch_weights/epoch-001.pt').exists()
                        if args.stage != 'ctc':
                            checks = [json.loads(line) for line in (output / 'smoke_checks.jsonl').open()]
                            assert {row['phase'] for row in checks} == {'joint', 'frozen'}
                    result = dict(stage=args.stage, phase=args.phase, passed=passed, step=trainer.state.global_step,
                        data=metadata, source_sha256=sources, metrics=metrics, checkpoint=str(checkpoint),
                        output_classes=classes, peak_cuda_GiB=torch.cuda.max_memory_allocated() / 2 ** 30,
                        evaluation_uploads=False)
                    write_json(output / 'RESULTS.json', result)
                    if passed:
                        write_json(output / 'PASS.json', result)
                        return 0
                    return 2 if complete else 75
                if not complete:
                    return 75
        finally:
            run.finish()
        base.final_benchmarks(model, rows, processor, tokenizer, output, config, topology)
        require_gate(args.stage, 'full', sources, metadata)
        return 0


def tiny_model():
    return ctc.CTCModel(Wav2Vec2Config(vocab_size=500, pad_token_id=0, hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_stride=(2, 2), conv_kernel=(3, 3),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2, feat_extract_norm='layer',
        hidden_dropout=0., attention_dropout=0., activation_dropout=0., feat_proj_dropout=0.,
        final_dropout=0., layerdrop=0., apply_spec_augment=False,
        ctc_loss_reduction='mean', ctc_zero_infinity=False))


def check():
    torch.set_num_threads(2)
    base.set_seed(42)
    metadata, tokenizer, rows = load_data()
    topology = topology_for('spottc_ce', tokenizer)
    assert len(topology['labels']) == 1498
    config = Wav2Vec2Config.from_pretrained(str(BACKBONE), local_files_only=True, vocab_size=500)
    assert read_json_string(config.to_dict()) == config.to_dict()
    for items in rows.values():
        for row in items:
            targets = row['labels']
            states = [state for token in targets for state in topology['mapping'][token]]
            assert base.content_ids(states, topology) == targets
            frames = round(row['duration'] * 16000)
            for kernel, stride in zip(config.conv_kernel, config.conv_stride):
                frames = (frames - kernel) // stride + 1
            required = len(targets) + sum(left == right for left, right in zip(targets, targets[1:]))
            assert frames >= required, (row['utt'], frames, required)
    token = tokenizer.encode('she')
    assert base.decode_path(token + [0] + token, tokenizer) == 'she she'
    states = [state for index in token for state in topology['mapping'][index]]
    assert base.decode_path(states * 2, tokenizer, topology) == 'she she'
    model = tiny_model().eval()
    inputs = torch.randn(2, 128)
    mask = torch.ones(2, 128, dtype=torch.long)
    mask[1, 96:] = 0
    labels = torch.tensor([[4, 4, 5], [6, 7, -100]])
    output = model(inputs, attention_mask=mask, labels=labels)
    lengths = model._get_feat_extract_output_lengths(mask.sum(-1)).long()
    target_lengths = (labels >= 0).sum(-1)
    expected = torch.nn.functional.ctc_loss(output.logits.log_softmax(-1).transpose(0, 1),
        labels[labels >= 0], lengths, target_lengths, blank=0, reduction='none')
    torch.testing.assert_close(output.loss, (expected / target_lengths).mean())
    assert output.logits.shape[-1] == 500
    output.loss.backward()
    gradient = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    for index in range(2):
        (model(inputs[index:index + 1], attention_mask=mask[index:index + 1],
               labels=labels[index:index + 1]).loss / 2).backward()
    torch.testing.assert_close(model.lm_head.weight.grad, gradient, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(gradient).all() and bool(gradient.abs().sum() > 0)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        converted = convert_to_fp32(model(inputs, attention_mask=mask))
    assert converted.hidden_states[1].sum(-1).tolist() == lengths.tolist()
    for invalid in [torch.tensor([[0]]), torch.full((1, 17), 4)]:
        try:
            model(inputs[:1], attention_mask=mask[:1], labels=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid CTC labels accepted')
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        restored = ctc.CTCModel.from_pretrained(directory).eval()
        torch.testing.assert_close(restored(inputs, attention_mask=mask).logits,
                                   model(inputs, attention_mask=mask).logits)
    check_resume(tokenizer)
    assert fit_passed(dict(WER_percent=10, CER_percent=5))
    assert not fit_passed(dict(WER_percent=10, CER_percent=6))
    records = [dict(step=step, WER_percent=80, CER_percent=60) for step in [511, 584, 657]]
    assert learning_passed(records, dict(WER_percent=100, CER_percent=100))
    assert not learning_passed(records[:2], dict(WER_percent=100, CER_percent=100))
    assert not learning_passed(records, dict(WER_percent=100, CER_percent=50))
    print('PASS: frozen BPE500 data, CTC feasibility/loss/gradients/masks/save-load, S3T2 1498-state roundtrip and gates')


def check_resume(tokenizer):
    processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
        padding_value=0., do_normalize=True, return_attention_mask=True)
    data = base.Dataset.from_list([dict(input_values=base.np.linspace(-1, 1, 128, dtype=base.np.float32),
                                      labels=[4, 5]) for _ in range(6)])
    config = dict(source_sha256={}, data=dict(tokenizer_sha256='CPU check'), phase='smoke')
    run = SimpleNamespace(log=lambda record: None)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        def trainer(stop=None):
            base.set_seed(42)
            arguments = base.TrainingArguments(output_dir=directory, use_cpu=True,
                per_device_train_batch_size=2, per_device_eval_batch_size=1, num_train_epochs=1,
                learning_rate=1e-4, save_strategy='epoch', evaluation_strategy='epoch',
                logging_steps=1, logging_nan_inf_filter=False, report_to=[],
                remove_unused_columns=False, disable_tqdm=True, save_safetensors=True)
            return CTCTrainer(model=tiny_model(), args=arguments, train_dataset=data, eval_dataset=data,
                data_collator=base.ChunkedCollator(base.Collator(processor), 1),
                compute_metrics=base.compute_metrics(tokenizer), tokenizer=processor,
                callbacks=[Policy(output, config, float('inf'), run, stop)])
        first = trainer(1)
        first.train()
        checkpoint = latest_checkpoint(output)
        assert checkpoint is not None and first.state.global_step == 1
        (output / 'checkpoint-999').mkdir()
        assert latest_checkpoint(output) == checkpoint
        resumed = trainer()
        resumed.train(resume_from_checkpoint=str(checkpoint))
        assert resumed.state.global_step == 3 and resumed.state.epoch == 1
        assert latest_checkpoint(output).name == 'checkpoint-3'
        assert (output / 'best_dev.pt').exists() and (output / 'epoch_weights/epoch-001.pt').exists()
        assert (output / 'checkpoint-999').exists()
        policy = Policy(output, config, float('inf'), run)
        for epoch in range(2, 11):
            policy.on_epoch_end(None, SimpleNamespace(epoch=epoch, global_step=epoch * 3), None,
                                model=resumed.model)
        members = sorted((output / 'epoch_weights').glob('*.pt'))
        for count in [5, 10]:
            state, _ = base.average(members[-count:], mfa=True)
            resumed.model.load_state_dict(state, strict=True)
    print('PASS: CPU optimizer/scheduler resume, incomplete-checkpoint rejection, epoch archives and last5/last10 averaging')


def read_json_string(value):
    return json.loads(json.dumps(value))


def stage_status():
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for phase in ['smoke', 'overfit', 'full']:
            output = HERE / phase / stage
            state = read_json(output / 'LATEST_STATE.json') if (output / 'LATEST_STATE.json').exists() else {}
            result[stage][phase] = dict(step=state.get('step', 0), epoch=state.get('epoch', 0),
                gate_recorded=gate_path(stage, phase).exists(),
                benchmark_complete=(output / 'BENCHMARK_COMPLETE.json').exists())
    return result


def run_stage(args):
    assert os.environ.get('PBS_JOBID') and torch.cuda.is_available(), 'Allocated PBS GPU required'
    assert 180 <= args.max_seconds <= 6000, 'Two-hour job budget must retain its 20-minute reserve'
    sources = source_hashes()
    submissions = read_json(HERE / 'SUBMISSIONS.json') if (HERE / 'SUBMISSIONS.json').exists() else []
    matches = [row for row in submissions if row['job'].split('.')[0] == os.environ['PBS_JOBID'].split('.')[0]]
    if matches:
        assert matches[-1]['source_sha256'] == sources, 'Submitted source contract changed'
    metadata, _, _ = load_data()
    stage_index = STAGES.index(args.stage)
    if stage_index:
        require_gate(STAGES[stage_index - 1], 'full', sources, metadata)
    jobs = HERE / 'jobs'
    jobs.mkdir(exist_ok=True)
    with (jobs / (args.stage + '.pipeline.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + args.max_seconds
        for phase in ['smoke', 'overfit', 'full']:
            if phase != 'full' and gate_path(args.stage, phase).exists():
                require_gate(args.stage, phase, sources, metadata)
                continue
            if phase == 'full' and (HERE / phase / args.stage / 'BENCHMARK_COMPLETE.json').exists():
                require_gate(args.stage, phase, sources, metadata)
                continue
            stops = [1, None] if phase == 'smoke' and latest_checkpoint(HERE / phase / args.stage) is None else [None]
            for stop in stops:
                remaining = int(deadline - time.monotonic())
                if remaining < 180:
                    print('PAUSED: budget exhausted; submit the same stage to resume.', flush=True)
                    return 75
                command = [sys.executable, '-u', str(HERE / 'train.py'), '--stage', args.stage,
                           '--phase', phase, '--max-seconds', str(remaining)]
                if stop is not None:
                    command += ['--stop-after-steps', str(stop)]
                print(f'RUN stage={args.stage} phase={phase} budget={remaining}s stop={stop}', flush=True)
                result = subprocess.run(command)
                write_json(jobs / (args.stage + '_latest.json'), dict(stage=args.stage, phase=phase,
                    job=os.environ['PBS_JOBID'], exit_code=result.returncode,
                    status='passed' if result.returncode == 0 else ('paused' if result.returncode == 75 else 'failed'),
                    timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), evaluation_uploads=False))
                if stop == 1 and result.returncode == 75:
                    continue
                if result.returncode != 0:
                    print('Stage did not complete; later stages are not started.', flush=True)
                    return result.returncode
        print(f'COMPLETE: {args.stage}; no automatic submission of later stages.', flush=True)
        return 0


def submit(args):
    assert 180 <= args.max_seconds <= 6000
    metadata, _, _ = load_data()
    sources = source_hashes()
    stage_index = STAGES.index(args.stage)
    if stage_index:
        require_gate(STAGES[stage_index - 1], 'full', sources, metadata)
    jobs = HERE / 'jobs'
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    log = jobs / f'{args.stage}_{stamp}.log'
    command = [shutil.which('qsub') or 'qsub', '-N', 'b500_' + args.stage,
               '-v', f'BPE_STAGE={args.stage},BPE_SECONDS={args.max_seconds}',
               '-o', str(log), str(HERE / 'run.pbs')]
    if args.dry_run:
        print(json.dumps(dict(command=command, stage=args.stage, data='TIMIT 4620/400/192',
            phases=['smoke', 'overfit', 'full'], evaluation_uploads=False), indent=2))
        return
    jobs.mkdir(exist_ok=True)
    with (jobs / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = HERE / 'SUBMISSIONS.json'
        records = read_json(path) if path.exists() else []
        queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
        live = {line.split()[0].split('.')[0] for line in queue.splitlines()
                if line.split() and line.split()[0].split('.')[0].isdigit()}
        active = [row for row in records if row['job'].split('.')[0] in live]
        if active:
            print(json.dumps(dict(already_active=active[-1]['job'], stage=active[-1]['stage']), indent=2))
            return
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job = result.stdout.strip()
        assert job.split('.')[0].isdigit(), f'Unexpected qsub output: {job}'
        record = dict(stage=args.stage, job=job, command=command, log=str(log),
            source_sha256=sources, data_metadata_sha256=base.sha256(DATA / 'metadata.json'),
            timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), evaluation_uploads=False)
        write_json(path, records + [record])
        print(json.dumps(dict(submitted=job, stage=args.stage, log=str(log)), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--check', action='store_true')
    operation.add_argument('--status', action='store_true')
    operation.add_argument('--submit', action='store_true')
    operation.add_argument('--run-stage', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stage', choices=STAGES, default='ctc')
    parser.add_argument('--phase', choices=['smoke', 'overfit', 'full'], default='smoke')
    parser.add_argument('--max-seconds', type=int, default=6000)
    parser.add_argument('--stop-after-steps', type=int)
    args = parser.parse_args()
    if args.dry_run and not args.submit:
        parser.error('--dry-run requires --submit')
    if args.check:
        check()
    elif args.status:
        print(json.dumps(stage_status(), indent=2))
    elif args.submit:
        submit(args)
    elif args.run_stage:
        sys.exit(run_stage(args))
    else:
        sys.exit(run_training(args))


if __name__ == '__main__':
    main()