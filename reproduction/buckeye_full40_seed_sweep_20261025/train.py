"""Independent physical-batch64 full40 training retaining dev-best three weights."""
import argparse
import fcntl
import gc
import json
import math
import time
import tempfile
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainingArguments, Wav2Vec2Processor, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler

from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.libri100_timit_seed_sweep_20261022.train import (
    QuietTrainer, ChunkedCollator, compute_metrics, start_wandb, append,
)
from reproduction.timit_seed_sweep_20261022.common import atomic_weights, snapshot_sources


class Trainer(QuietTrainer):
    def _get_train_sampler(self):
        return LengthGroupedSampler(batch_size=self.args.train_batch_size, lengths=self.train_dataset.lengths)


class Policy(TrainerCallback):
    def __init__(self, output, config, deadline, run):
        self.output, self.config, self.deadline, self.run = output, config, deadline, run

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if abs(state.epoch - round(state.epoch)) > 1e-6:
            return control
        value = metrics['eval_nosil_PER_percent']
        assert math.isfinite(value)
        path = self.output / 'epoch_weights' / f'epoch-{round(state.epoch):03d}.pt'
        metadata = dict(epoch=round(state.epoch), step=state.global_step, recipe=self.config['name'],
                        seed=self.config['seed'], dev_PER=value, model_config=model.config.to_dict(),
                        source_sha256=self.config['source_sha256'])
        index = self.output / 'EPOCHS.json'
        previous = json.loads(index.read_text()) if index.exists() else []
        if any(item['epoch'] == metadata['epoch'] for item in previous):
            return control
        candidate = dict(path=str(path), **metadata)
        members = sorted(previous + [candidate], key=lambda item: (item['dev_PER'], item['epoch']))[:3]
        append(self.output / 'epoch_selection.jsonl', dict(epoch=metadata['epoch'], step=metadata['step'],
               dev_PER=value, retained=candidate in members))
        if candidate not in members:
            return control
        atomic_weights(model, path, metadata)
        candidate['sha256'] = common.base.sha(path)
        common.base.write(index, members)
        common.base.write(self.output / 'BEST_DEV.json', members[0])
        for item in previous:
            if item not in members:
                obsolete = path.parent / Path(item['path']).name
                assert str(obsolete) == item['path'] and not obsolete.is_symlink()
                obsolete.unlink()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = dict(logs or {}, optimizer_step=state.global_step, epoch=state.epoch)
        append(self.output / 'history.jsonl', record)
        self.run.log(record)


def profile(config, data, rows, vocab, processor, output):
    from reproduction.buckeye_seed_sweep_20261022.profile_recipe import attempt
    assert common.base.sha(common.base.SSL / 'pytorch_model.bin') == common.base.SSL_SHA
    records = attempt(config, data, rows, vocab, processor, output, 64)
    model, topo = common.base.build_model(config, vocab, math.ceil(len(data['train']) / 64))
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    model.cuda().eval()
    from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
    from reproduction.timit_seed_sweep_20261022.pfer import score_predictions
    for split in ('dev', 'timit_dev'):
        target = output / f'{split}_probe.json'
        summary = evaluate(config, model, rows[split][:2], vocab, processor, topo, target)
        score_predictions(target.with_suffix('.predictions.jsonl'), summary)
        assert summary['modes']['forced_alignment']['alignment_failures'] == 0
    del model
    gc.collect()
    torch.cuda.empty_cache()
    common.base.write(output / 'PASS.json', dict(config=config, batches=records, physical_batch=64,
                      parameter_dtype='float32', saved_weights_dtype='float32', evaluation_uploads=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=36000)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    rows, vocab, contracts, data_root = common.checked_rows()
    config = common.recipe(args.recipe, contracts)
    assert config == json.loads((common.HERE / 'planned_configs' / f'{args.recipe}.json').read_text())
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    data = {split: common.AudioDataset(rows[split], processor) for split in ('train', 'dev')}
    output = common.HERE / ('profiles' if args.profile else 'models') / args.recipe
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot_sources(output, config['source_sha256'])
    if args.profile:
        profile(config, data, rows, vocab, processor, output)
        return
    contract = output / 'run_config.json'
    if contract.exists():
        assert json.loads(contract.read_text()) == config
    else:
        common.base.write(contract, config)
    set_seed(config['seed'])
    model, topo = common.base.build_model(config, vocab, math.ceil(len(data['train']) / 64))
    assert topo is None and all(parameter.dtype == torch.float32 for parameter in model.parameters())
    with tempfile.TemporaryDirectory(dir=output, prefix='initial_dtype_') as directory:
        weights = Path(directory) / 'weights.pt'
        atomic_weights(model, weights, dict(recipe=config['name'], initialization_only=True))
        saved = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
        assert all(tensor.dtype == torch.float32 for tensor in saved['model'].values() if tensor.is_floating_point())
        del saved
    common.base.write(output / 'TRAINABLE_PARAMETERS.json', common.base.parameter_manifest(model))
    processor.save_pretrained(output / 'processor')
    training_args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=64,
        per_device_eval_batch_size=64, gradient_accumulation_steps=1, num_train_epochs=40,
        learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'], weight_decay=.005,
        bf16=True, fp16=False, seed=config['seed'], data_seed=config['seed'], evaluation_strategy='epoch',
        save_strategy='epoch', save_total_limit=1, load_best_model_at_end=False, logging_steps=10,
        report_to=[], dataloader_num_workers=4, dataloader_pin_memory=True, remove_unused_columns=False,
        group_by_length=True, max_grad_norm=1., disable_tqdm=True, save_safetensors=True)
    assert training_args.world_size == 1
    run = start_wandb(output, config, 'training')
    trainer = Trainer(model=model, args=training_args, train_dataset=data['train'], eval_dataset=data['dev'],
        data_collator=ChunkedCollator(common.base.Collator(processor), 64), tokenizer=processor,
        compute_metrics=compute_metrics(vocab, topo),
        callbacks=[Policy(output, config, time.monotonic() + args.max_seconds, run)])
    assert trainer.accelerator.num_processes == 1
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda path: int(path.name.split('-')[1]))
    if not (output / 'TRAIN_COMPLETE.json').exists():
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None,
                      ignore_keys_for_eval=['hidden_states', 'attentions'])
        complete = trainer.state.epoch >= config['epochs'] - 1e-5
        common.base.write(output / ('TRAIN_COMPLETE.json' if complete else 'PAUSED.json'),
                          dict(epoch=trainer.state.epoch, step=trainer.state.global_step, complete=complete))
    else:
        complete = True
    run.summary['training_complete'] = complete
    run.finish()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    if not complete:
        raise SystemExit(75)


if __name__ == '__main__':
    main()