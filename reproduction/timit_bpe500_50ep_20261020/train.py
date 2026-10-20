"""Matched 50-epoch TIMIT BPE500 recognition and segmentation experiments."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from reproduction.timit_bpe500_stages_20261020.ottc_ce_physical64_10ep import train as ottc
from reproduction.checkpoint_retention import CheckpointRetentionCallback
from reproduction.timit_bpe500_50ep_20261020 import segmentation

parent = ottc.parent
physical = ottc.physical
base = parent.base
torch = parent.torch
RECIPES = ('ctc', 'ottc_ce', 'ottc_jsd', 'spottc_ce', 'spottc_jsd')
PARENT_SOURCES = ottc.source_hashes()


def source_hashes():
    sources = dict(PARENT_SOURCES)
    for path in [*HERE.glob('*.py'), *HERE.glob('*.pbs'), ROOT / 'reproduction/checkpoint_retention.py',
                 Path(segmentation.scorer.__file__), ROOT / 'reproduction/native_stride_ottc_20261018/evaluate.py']:
        sources[str(path.relative_to(ROOT))] = base.sha256(path)
    return sources


def topology_for(recipe, tokenizer):
    return base.build_topology(tokenizer, 'spottc_s3t2') if recipe.startswith('spottc') else None


def configuration(recipe, metadata, rows, identities):
    assert recipe in RECIPES
    steps = math.ceil(len(rows['train']) / 64)
    return dict(name=HERE.name, recipe=HERE.name + '__' + recipe, stage=recipe, phase='full',
        epochs=50, global_batch=64, physical_train_batch=64, microbatch=64, gradient_accumulation_steps=1,
        optimizer_steps_per_epoch=steps, planned_optimizer_steps=50 * steps,
        seed=42, learning_rate=2e-4, warmup_steps=500, weight_decay=.005, spec_augment=True,
        objective='ctc' if recipe == 'ctc' else ('jsd' if recipe.endswith('jsd') else 'crossentropy'),
        output_classes=1498 if recipe.startswith('spottc') else 500,
        topology='s3_t2' if recipe.startswith('spottc') else 'none',
        topology_adaptation='Three ordered states per nonblank ID, including unused special IDs; references use IDs 4..499.',
        freeze_alignment_after_epochs=30, frozen_steps_for_regfrozen=30 * steps + 1,
        data=metadata, source_sha256=source_hashes(), **identities,
        initialization='Fresh matched official Wav2Vec2-large-LV60 SSL initialization; no 10/40-epoch checkpoints imported.',
        scheduler='Linear warmup/decay over 3650 steps, not a continuation of a zero-LR ten-epoch schedule.',
        batch_protocol='Exactly one forward/backward per physical64 batch, final batch12, no accumulation or OOM fallback.',
        stopping='Exactly 50 epochs unless numerical failure or resumable walltime pause; no recognition gate or early stopping.',
        checkpoint_policy='Atomic named epoch weights; retain dev-best3 and last10 plus evaluation dependencies, best-dev and latest2 full states.',
        decoding='Audio-only greedy; repeat collapse and blank removal; existing state-reset decoder for S3T2; no LM.',
        segmentation=segmentation.PROTOCOL,
        evaluation_uploads=False)


class WholeBatchCollator:
    def __init__(self, processor, topology=None):
        self.collator = base.Collator(processor, topology['mapping'] if topology else None)

    def __call__(self, examples):
        return dict(microbatches=[self.collator(examples)], logical_batch_size=len(examples))


def training_arguments(output, config, cpu=False):
    return base.TrainingArguments(output_dir=str(output), per_device_train_batch_size=config['physical_train_batch'],
        per_device_eval_batch_size=2, gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_train_epochs=config['epochs'], learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'], bf16=not cpu, use_cpu=cpu, seed=42, data_seed=42,
        evaluation_strategy='epoch', save_strategy='epoch', save_total_limit=None, logging_steps=10,
        logging_nan_inf_filter=False, report_to=[], dataloader_num_workers=2,
        remove_unused_columns=False, group_by_length=True, length_column_name='length',
        max_grad_norm=1., disable_tqdm=True, save_safetensors=True)


def build_model(config):
    if config['stage'] == 'ctc':
        return physical.build_model(config)
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
    model = wvForOTTC(wvOTTCConfig(wav2vec2_path=str(parent.BACKBONE), vocab_size=config['output_classes'],
        hidden_size=1024, bins_metric=config['objective'], forward_type='regfrozen', pad_token_id=0,
        frozen_steps_for_regfrozen=config['frozen_steps_for_regfrozen']))
    model.wav2vec2_freeze.requires_grad_(False)
    return model


class PhysicalOTTCTrainer(ottc.PhysicalOTTCTrainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if abs(model.step - model.frozen_steps_for_regfrozen) <= 2:
            parent.bpe.checked.check_gradients(model)
        return loss


def prepare_references(rows):
    references = segmentation.load_references(rows)
    path = HERE / 'data/word_references.json'
    if path.exists():
        assert parent.read_json(path) == references, 'Human word reference contract changed'
    else:
        parent.write_json(path, references)
    return references, base.sha256(path)


def identities(reference_hash):
    return dict(backbone_sha256=base.sha256(parent.BACKBONE / 'pytorch_model.bin'),
        backbone_config_sha256=base.sha256(parent.BACKBONE / 'config.json'),
        word_references_sha256=reference_hash)


def evaluate(model, rows, processor, tokenizer, topology, references, destination):
    metrics = base.evaluate(model, rows, processor, tokenizer, destination, 2, topology)
    metrics['segmentation'] = segmentation.score_predictions(destination, references, tokenizer, topology)
    parent.write_json(destination.with_suffix('.metrics.json'), metrics)
    return metrics


class Policy(base.Policy):
    def __init__(self, *args, evaluation, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation = evaluation

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, model=model, metrics=metrics, **kwargs)
        rows, processor, tokenizer, topology, references = self.evaluation
        destination = self.output / 'development' / f'step-{state.global_step:06d}.jsonl'
        result = evaluate(model, rows, processor, tokenizer, topology, references, destination)
        parent.write_json(self.output / 'LATEST_DEVELOPMENT.json', dict(epoch=state.epoch,
            step=state.global_step, recognition={key: result[key] for key in ['WER_percent', 'CER_percent']},
            segmentation=result['segmentation'], evaluation_uploads=False))
        return control


def make_trainer(model, rows, processor, tokenizer, topology, references, output, config, tracking, deadline):
    dataset = {split: base.Dataset.from_list([dict(wav=row['wav'], length=round(row['duration'] * 16000),
        labels=row['labels']) for row in rows[split]]) for split in ['train', 'dev']}
    trainer_class = physical.PhysicalCTCTrainer if config['stage'] == 'ctc' else PhysicalOTTCTrainer
    return trainer_class(model=model, args=training_arguments(output, config),
        train_dataset=dataset['train'], eval_dataset=dataset['dev'], tokenizer=processor,
        data_collator=WholeBatchCollator(processor, topology), compute_metrics=base.compute_metrics(tokenizer, topology),
        callbacks=[Policy(output, config, deadline, tracking,
            evaluation=(rows['dev'], processor, tokenizer, topology, references['dev'])),
            CheckpointRetentionCallback(output)])


def final_benchmarks(model, rows, processor, tokenizer, topology, references, output, config):
    complete = parent.read_json(output / 'TRAIN_COMPLETE.json')
    assert complete['epoch'] >= 50 and complete['step'] == 3650
    best = parent.read_json(output / 'BEST_DEV.json')
    members = sorted((output / 'epoch_weights').glob('epoch-*.pt'))
    candidates = {'best_dev': dict(path=str(output / 'best_dev.pt'), step=best['step'],
                                   sha256=base.sha256(output / 'best_dev.pt'))}
    for count in [5, 10]:
        selected = members[-count:]
        records = [dict(path=str(path), **parent.read_json(path.with_suffix('.json'))) for path in selected]
        assert [record['epoch'] for record in records] == list(range(51 - count, 51))
        candidates['last' + str(count)] = dict(members=records, step=3650)
    result = dict(candidates={}, protocol=config, evaluation_uploads=False)
    for split in ['dev', 'test']:
        for name, candidate in candidates.items():
            if name == 'best_dev':
                state = torch.load(candidate['path'], map_location='cpu', weights_only=False, mmap=True)['model']
            else:
                state, _ = base.average([Path(member['path']) for member in candidate['members']], mfa=True)
            model.load_state_dict(state, strict=True)
            del state
            model.step = candidate['step'] - 1
            result['candidates'].setdefault(name, dict(provenance=candidate))[split] = evaluate(model,
                rows[split], processor, tokenizer, topology, references[split],
                output / 'evaluation' / name / (split + '.jsonl'))
            parent.write_json(output / 'RESULTS.json', result)
        if split == 'dev':
            result['selected_by_dev_WER'] = min(candidates, key=lambda name: result['candidates'][name]['dev']['WER_percent'])
            result['selected_by_dev_Rvalue20'] = max(candidates,
                key=lambda name: result['candidates'][name]['dev']['segmentation']['boundaries']['20']['r_value_pct'])
            parent.write_json(output / 'DEV_SELECTION.json', dict(recognition=result['selected_by_dev_WER'],
                segmentation=result['selected_by_dev_Rvalue20'], test_used=False))
    parent.write_json(output / 'RESULTS.json', result)
    parent.write_json(output / 'BENCHMARK_COMPLETE.json', dict(epoch=50, step=3650,
        selected_by_dev_WER=result['selected_by_dev_WER'], selected_by_dev_Rvalue20=result['selected_by_dev_Rvalue20'],
        results_sha256=base.sha256(output / 'RESULTS.json'), evaluation_uploads=False))


def run(recipe, max_seconds):
    assert os.environ.get('PBS_JOBID') and torch.cuda.is_available(), 'Allocated PBS GPU required'
    assert 180 <= max_seconds <= 6000
    os.environ['OTTC_EVAL_UPLOAD'] = '0'
    started = time.monotonic()
    torch.set_num_threads(4)
    base.set_seed(42)
    metadata, tokenizer, rows = parent.load_data()
    references, reference_hash = prepare_references(rows)
    config = configuration(recipe, metadata, rows, identities(reference_hash))
    submissions = parent.read_json(HERE / 'SUBMISSIONS.json') if (HERE / 'SUBMISSIONS.json').exists() else []
    for record in submissions:
        if record['job'].split('.')[0] == os.environ['PBS_JOBID'].split('.')[0]:
            assert record['recipe'] == recipe and record['source_sha256'] == config['source_sha256']
    output = HERE / 'models' / recipe
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = output / 'run_config.json'
        if contract.exists():
            assert parent.read_json(contract) == config, 'Resume/source/schedule contract changed'
        else:
            parent.write_json(contract, config)
            parent.write_json(output / 'word_references.json', references)
            for relative in config['source_sha256']:
                target = output / 'source_snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
        if (output / 'BENCHMARK_COMPLETE.json').exists():
            return 0
        topology = topology_for(recipe, tokenizer)
        if topology:
            parent.write_json(output / 'state_labels.json', topology['labels'])
            parent.write_json(output / 'token_to_states.json', topology['mapping'])
        model = build_model(config)
        model.freeze_feature_encoder()
        processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
            padding_value=0., do_normalize=True, return_attention_mask=True)
        processor.save_pretrained(output / 'processor')
        resume = parent.latest_checkpoint(output)
        if resume:
            assert base.sha256(resume / 'model.safetensors') == parent.read_json(resume / 'COMPLETE.json')['model_sha256']
            parent.write_json(output / 'RESUME_ORIGIN.json', dict(checkpoint=str(resume),
                trainer_state_sha256=base.sha256(resume / 'trainer_state.json')))
        model.cuda()
        if not (output / 'INITIAL_DEV.json').exists():
            assert resume is None, 'Missing initialization metrics for a resumed run'
            initial = evaluate(model, rows['dev'], processor, tokenizer, topology, references['dev'],
                               output / 'initial/dev.jsonl')
            parent.write_json(output / 'INITIAL_DEV.json', initial)
        tracking = parent.start_tracking(output, config)
        trainer = make_trainer(model, rows, processor, tokenizer, topology, references, output, config,
                               tracking, started + max_seconds)
        print(json.dumps(dict(recipe=recipe, epochs=50, optimizer_steps=3650, physical_batch=64,
                              gradient_accumulation_steps=1, warmup_steps=500)), flush=True)
        try:
            if (output / 'TRAIN_COMPLETE.json').exists():
                assert resume is not None
                trainer._load_from_checkpoint(str(resume))
            else:
                trainer.train(resume_from_checkpoint=str(resume) if resume else None,
                              ignore_keys_for_eval=['hidden_states', 'attentions'])
                checkpoint = parent.latest_checkpoint(output)
                assert checkpoint is not None
                complete = trainer.state.epoch >= 50 - 1e-5
                state = dict(epoch=trainer.state.epoch, step=trainer.state.global_step,
                    checkpoint=str(checkpoint), complete=complete, job=os.environ['PBS_JOBID'])
                parent.write_json(output / 'LATEST_STATE.json', state)
                if not complete:
                    print('PAUSED: full resume state saved; 50-epoch target unchanged.', flush=True)
                    return 75
                assert trainer.state.global_step == 3650
                parent.write_json(output / 'TRAIN_COMPLETE.json', state)
        finally:
            tracking.finish()
        final_benchmarks(model, rows, processor, tokenizer, topology, references, output, config)
        return 0


def submission_command(recipe, log, dependency=None):
    command = [shutil.which('qsub') or 'qsub', '-N', 'b50_' + recipe, '-v', 'BPE_RECIPE=' + recipe,
               '-o', str(log)]
    if dependency:
        command += ['-W', 'depend=afterok:' + dependency]
    return command + [str(HERE / 'run.pbs')]


def submit(recipes, dry_run=False):
    metadata, _, rows = parent.load_data()
    references, reference_hash = prepare_references(rows)
    identity = identities(reference_hash)
    jobs = HERE / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    with (jobs / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = HERE / 'SUBMISSIONS.json'
        records = parent.read_json(path) if path.exists() else []
        queue = subprocess.run(['qstat'], capture_output=True, text=True, check=True).stdout
        live = {line.split()[0].split('.')[0] for line in queue.splitlines()
                if line.split() and line.split()[0].split('.')[0].isdigit()}
        dependency = None
        for recipe in recipes:
            config = configuration(recipe, metadata, rows, identity)
            output = HERE / 'models' / recipe
            if (output / 'BENCHMARK_COMPLETE.json').exists():
                assert parent.read_json(output / 'run_config.json') == config
                print(json.dumps(dict(recipe=recipe, completed=True)), flush=True)
                continue
            existing = [record for record in records if record['recipe'] == recipe and record['job'].split('.')[0] in live]
            if existing:
                assert existing[-1]['source_sha256'] == config['source_sha256']
                dependency = existing[-1]['job']
                print(json.dumps(dict(recipe=recipe, already_active=dependency)), flush=True)
                continue
            log = jobs / (recipe + '_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '.log')
            command = submission_command(recipe, log, dependency)
            if dry_run:
                print(json.dumps(dict(recipe=recipe, command=command, epochs=50, steps=3650,
                    physical_batch=64, evaluation_uploads=False)), flush=True)
                dependency = 'PREVIOUS_' + recipe
                continue
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job = result.stdout.strip()
            assert job.split('.')[0].isdigit(), job
            record = dict(recipe=recipe, job=job, command=command, depends_on=dependency, log=str(log),
                epochs=50, physical_batch=64, source_sha256=config['source_sha256'],
                data_metadata_sha256=base.sha256(parent.DATA / 'metadata.json'), word_references_sha256=reference_hash,
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), evaluation_uploads=False)
            records.append(record)
            parent.write_json(path, records)
            dependency = job
            print(json.dumps(dict(submitted=job, recipe=recipe, depends_on=record['depends_on'], epochs=50)), flush=True)


def check():
    parent.check()
    metadata, tokenizer, rows = parent.load_data()
    assert len(rows['train']) == 4620 and len(rows['dev']) == 400 and len(rows['test']) == 192
    processor = base.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,
        padding_value=0., do_normalize=True, return_attention_mask=True)
    examples = [dict(input_values=base.np.linspace(-1, 1, 128, dtype=base.np.float32),
                     labels=tokenizer.encode('she she')) for _ in range(64)]
    with tempfile.TemporaryDirectory() as directory:
        for recipe in RECIPES:
            config = configuration(recipe, metadata, rows, {})
            args = training_arguments(Path(directory) / recipe, config, cpu=True)
            assert args.num_train_epochs == 50 and config['planned_optimizer_steps'] == 3650
            assert args.per_device_train_batch_size == 64 and args.gradient_accumulation_steps == 1
            assert args.warmup_steps == 500 and config['frozen_steps_for_regfrozen'] == 2191
            topology = topology_for(recipe, tokenizer)
            assert config['output_classes'] == (len(topology['labels']) if topology else 500)
            collator = WholeBatchCollator(processor, topology)
            for size in [64, 12]:
                batch = collator(examples[:size])
                assert len(batch['microbatches']) == 1 and batch['logical_batch_size'] == size
                labels = batch['microbatches'][0]['labels'][0].tolist()
                assert base.content_ids(labels, topology) == examples[0]['labels']
            command = submission_command('spottc_jsd', Path(directory) / 'job.log', '123.opbs')
            assert 'depend=afterok:123.opbs' in command and 'BPE_RECIPE=spottc_jsd' in command
            for previous_epoch in [10, 40, 49]:
                assert previous_epoch < configuration('ctc', metadata, rows, {})['epochs']
            segmentation.check(rows, tokenizer)
    print('PASS: all five recipes use 50 epochs/3650 steps, physical64, tail12, 500 warmup and original CE/JSD routing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--submit', action='store_true')
    modes.add_argument('--submit-all', action='store_true')
    parser.add_argument('--recipe', choices=RECIPES, default='ctc')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=6000)
    args = parser.parse_args()
    if args.dry_run and not (args.submit or args.submit_all):
        parser.error('--dry-run requires --submit or --submit-all')
    if args.check:
        check()
    elif args.submit or args.submit_all:
        submit(RECIPES if args.submit_all else [args.recipe], args.dry_run)
    else:
        sys.exit(run(args.recipe, args.max_seconds))


if __name__ == '__main__':
    main()