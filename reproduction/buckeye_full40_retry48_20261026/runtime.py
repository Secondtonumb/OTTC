"""Explicit runtime-only continuation of the frozen full40 recipe.

Keep 64 examples per optimizer update, but collate 48 + 16 for training and
48 for validation. Original recipe files, configs and source snapshots stay intact.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from transformers import TrainerCallback, TrainingArguments, Wav2Vec2Processor, set_seed

from reproduction.buckeye_full40_seed_sweep_20261025 import common, train
from reproduction.checkpoint_retention import verify_saved_checkpoint
from reproduction.timit_seed_sweep_20261022.common import snapshot_sources

HERE = Path(__file__).resolve().parent
MICRO = 48


def hashes():
    return {str(p): common.base.sha(p) for p in sorted(HERE.iterdir())
            if p.suffix in ('.py', '.sh', '.pbs')}


def patch_runtime():
    arguments, collator = train.TrainingArguments, train.ChunkedCollator

    def training_arguments(*args, **kwargs):
        kwargs['per_device_eval_batch_size'] = MICRO
        return arguments(*args, **kwargs)

    def chunked_collator(base, micro):
        assert micro == 64
        return collator(base, MICRO)

    train.TrainingArguments = training_arguments
    train.ChunkedCollator = chunked_collator


class ProbeCallback(TrainerCallback):
    """Always save after a probe step, including after loading old intervals."""
    def on_step_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        control.should_save = True
        return control


def gate(name):
    rows, vocab, contracts, data_root = common.checked_rows()
    config = common.recipe(name, contracts)
    output = HERE / 'gates' / name
    output.mkdir(parents=True, exist_ok=True)
    assert config == json.loads((common.HERE / 'models' / name / 'run_config.json').read_text())
    provenance = dict(runtime_sha256=hashes(), source_sha256=config['source_sha256'],
                      physical_batch=MICRO, logical_batch=64, evaluation_uploads=False)
    marker = output / 'PASS.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert all(saved[k] == v for k, v in provenance.items())
        return
    snapshot_sources(output, {**config['source_sha256'], **hashes()})
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    set_seed(config['seed'])
    torch.set_num_threads(4)
    # Exercise the two longest training batches and a long validation batch.
    longest = sorted(rows['train'], key=lambda row: row['duration'], reverse=True)[:128]
    dev = sorted(rows['dev'], key=lambda row: row['duration'], reverse=True)[:48]
    training_data, dev_data = common.AudioDataset(longest, processor), common.AudioDataset(dev, processor)
    evidence = []
    for limit, resume in ((2, None), (3, str(output / 'checkpoint-2'))):
        model, topo = common.base.build_model(config, vocab, math.ceil(len(rows['train']) / 64))
        assert topo is None and all(p.dtype == torch.float32 for p in model.parameters())
        args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=64,
            per_device_eval_batch_size=MICRO, max_steps=limit, gradient_accumulation_steps=1,
            learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'], weight_decay=.005,
            bf16=True, fp16=False, seed=config['seed'], data_seed=config['seed'],
            evaluation_strategy='steps', eval_steps=1, save_strategy='steps', save_steps=1,
            save_total_limit=1, logging_steps=1, report_to=[], dataloader_num_workers=4,
            remove_unused_columns=False, group_by_length=True, max_grad_norm=1.,
            disable_tqdm=True, save_safetensors=True)
        trainer = train.Trainer(model=model, args=args, train_dataset=training_data,
            eval_dataset=dev_data, data_collator=train.ChunkedCollator(common.base.Collator(processor), MICRO),
            tokenizer=processor, compute_metrics=train.compute_metrics(vocab, topo), callbacks=[ProbeCallback()])
        assert trainer.accelerator.num_processes == 1
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = trainer.train(resume_from_checkpoint=resume, ignore_keys_for_eval=['hidden_states', 'attentions'])
        assert trainer.state.global_step == limit and math.isfinite(result.training_loss)
        verify_saved_checkpoint(output, limit)
        evidence.append(dict(step=limit, resume_from=resume, seconds=time.monotonic() - started,
                             peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                             history=trainer.state.log_history))
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    from safetensors import safe_open
    with safe_open(str(output / 'checkpoint-3/model.safetensors'), framework='pt') as saved:
        assert all(saved.get_slice(k).get_dtype() == 'F32' for k in saved.keys()
                   if saved.get_slice(k).get_dtype() in ('F16', 'BF16', 'F32', 'F64'))
    common.base.write(marker, dict(**provenance, recipe=name, resume_verified=True,
        validation_to_training_verified=True, longest64_verified=True, saved_model_dtype='float32',
        job_id=os.environ.get('PBS_JOBID'), gpu=torch.cuda.get_device_name(), training=evidence))
    print('GATE_PASS', name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--stage', choices=('gate', 'train'), required=True)
    parser.add_argument('--max-seconds', type=int, default=28800)
    args = parser.parse_args()
    if args.stage == 'gate':
        gate(args.recipe)
        return
    marker = json.loads((HERE / 'gates' / args.recipe / 'PASS.json').read_text())
    assert marker['runtime_sha256'] == hashes()
    output = common.HERE / 'models' / args.recipe
    snapshot_sources(output, hashes())
    common.base.write(output / 'RUNTIME_BATCH48.json', dict(runtime_sha256=hashes(),
        physical_batch=MICRO, logical_batch=64, validation_batch=MICRO,
        chunks=[48, 16], resume_checkpoint='checkpoint-285',
        gate=str(HERE / 'gates' / args.recipe / 'PASS.json'), job_id=os.environ.get('PBS_JOBID'),
        evaluation_uploads=False, learning_rate_and_optimizer_schedule_unchanged=True))
    patch_runtime()
    sys.argv = [sys.argv[0], '--recipe', args.recipe, '--max-seconds', str(args.max_seconds)]
    train.main()


if __name__ == '__main__':
    main()
