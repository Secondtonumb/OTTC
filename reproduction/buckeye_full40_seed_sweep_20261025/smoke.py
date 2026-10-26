"""Real-data short training, checkpoint resume and dual-corpus evaluation gate."""
import argparse
import gc
import json
import math

import torch
from transformers import TrainingArguments, Wav2Vec2Processor, set_seed

from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.buckeye_full40_seed_sweep_20261025.train import Trainer, ChunkedCollator, compute_metrics
from reproduction.checkpoint_retention import verify_saved_checkpoint
from reproduction.timit_ssl_native10_20261019.evaluate import evaluate
from reproduction.timit_seed_sweep_20261022.pfer import score_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    args = parser.parse_args()
    rows, vocab, contracts, data_root = common.checked_rows()
    config = common.recipe(args.recipe, contracts)
    assert config == json.loads((common.HERE / 'planned_configs' / f'{args.recipe}.json').read_text())
    gate = json.loads((common.HERE / 'profiles' / args.recipe / 'PASS.json').read_text())
    assert gate['config'] == config and gate['physical_batch'] == 64
    output = common.HERE / 'smoke' / args.recipe
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config['seed'])
    torch.set_num_threads(4)
    processor = Wav2Vec2Processor.from_pretrained(data_root / 'processor')
    train_data = common.AudioDataset(rows['train'][:192], processor)
    dev_data = common.AudioDataset(rows['dev'][:2], processor)
    evidence = []
    for limit, resume in ((2, None), (3, str(output / 'checkpoint-2'))):
        model, topo = common.base.build_model(config, vocab, math.ceil(len(rows['train']) / 64))
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        training_args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=64,
            per_device_eval_batch_size=64, gradient_accumulation_steps=1, max_steps=limit,
            learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'], weight_decay=.005,
            bf16=True, fp16=False, seed=config['seed'], data_seed=config['seed'],
            evaluation_strategy='steps', eval_steps=limit, save_strategy='steps', save_steps=limit,
            save_total_limit=1, logging_steps=1, report_to=[], dataloader_num_workers=4,
            remove_unused_columns=False, group_by_length=True, max_grad_norm=1.,
            disable_tqdm=True, save_safetensors=True)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=dev_data,
            data_collator=ChunkedCollator(common.base.Collator(processor), 64), tokenizer=processor,
            compute_metrics=compute_metrics(vocab, topo))
        assert trainer.accelerator.num_processes == 1
        result = trainer.train(resume_from_checkpoint=resume, ignore_keys_for_eval=['hidden_states', 'attentions'])
        assert trainer.state.global_step == limit and math.isfinite(result.training_loss)
        verify_saved_checkpoint(output, limit)
        evidence.append(dict(step=limit, resumed_from=resume, train_loss=result.training_loss,
                             history=trainer.state.log_history))
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    from safetensors.torch import load_file
    saved = load_file(str(output / 'checkpoint-3/model.safetensors'))
    assert all(value.dtype == torch.float32 for value in saved.values() if value.is_floating_point())
    model, topo = common.base.build_model(config, vocab, 285)
    model.load_state_dict(saved)
    del saved
    model.cuda().eval()
    evaluation = {}
    for split in ('dev', 'timit_dev'):
        target = output / f'{split}_probe.json'
        summary = evaluate(config, model, rows[split][:2], vocab, processor, topo, target)
        pfer = score_predictions(target.with_suffix('.predictions.jsonl'), summary)
        assert summary['utterances'] == 2
        assert summary['modes']['forced_alignment']['alignment_failures'] == 0
        evaluation[split] = dict(summary=summary, pfer=pfer)
    common.base.write(output / 'PASS.json', dict(recipe=args.recipe, source_sha256=config['source_sha256'],
        expected_data=contracts, physical_batch=64, independent_world_size=1,
        saved_model_dtype='float32', resume_verified=True, training=evidence, evaluation=evaluation,
        evaluation_uploads=False, smoke_weights_used_for_formal_training=False))
    print('SMOKE_PASS', args.recipe, flush=True)


if __name__ == '__main__':
    main()