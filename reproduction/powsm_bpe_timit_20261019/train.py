"""TIMIT text recognition with POWSM tokens, OTTC/SPOTTC and upstream CE/JSD."""
import argparse
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT / 'reproduction/native_stride_ottc_20261018'
TOPOLOGY_VENDOR = ROOT / 'reproduction/zapformer_ifmdd_topology_ottc_20261013/vendor'
sys.path[:0] = [str(ROOT / 'reproduction/vendor'), str(ROOT), str(NATIVE), str(ROOT / 'reproduction')]

import numpy as np
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from datasets import Dataset
from rapidfuzz.distance import Levenshtein
from transformers import TrainerCallback, TrainingArguments, Wav2Vec2FeatureExtractor, set_seed
from transformers.trainer_utils import get_last_checkpoint

from common import Collator, atomic_weights
from reproduction.native_stride_ottc_20261018.train import ChunkedCollator, QuietTrainer
from reproduction.average_checkpoints import average
from reproduction.powsm_bpe_timit_20261019.prepare import Tokenizer, normalize_text, sha256, write_json
from reproduction.wandb_policy import init_evaluation


def source_hashes():
    files = [*HERE.glob('*.py'), *HERE.glob('*.pbs'), *sorted((ROOT / 'ottc').rglob('*.py')),
             NATIVE / 'common.py', NATIVE / 'train.py', ROOT / 'reproduction/average_checkpoints.py',
             ROOT / 'reproduction/wandb_policy.py', ROOT / 'reproduction/libri960_continue5_20261017/environment.sh',
             *sorted((TOPOLOGY_VENDOR / 'utils').glob('*.py'))]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def build_topology(tokenizer, head):
    if head == 'ottc':
        return None
    assert head == 'spottc_s3t2'
    sys.path.insert(0, str(TOPOLOGY_VENDOR))
    from utils.subphonetic_topology import make_subphone_labels, decode_subphone_ids
    labels = make_subphone_labels([f'bpe{index:05d}' for index in range(1, len(tokenizer.tokens))], 's3_t2')
    indices = {label: index for index, label in enumerate(labels)}
    mapping = {index: [indices[f'bpe{index:05d}_{state}'] for state in range(3)]
               for index in range(1, len(tokenizer.tokens))}
    assert len(labels) == 1 + 3 * (len(tokenizer.tokens) - 1)
    return dict(labels=labels, mapping=mapping, decode=decode_subphone_ids)


def content_ids(ids, topology):
    valid = [int(index) for index in ids if int(index) >= 0]
    if topology is None:
        return valid
    decoded = topology['decode'](valid, topology['labels'], mode='phone')
    return [int(label.removeprefix('bpe')) for label in decoded.split()]


def decode_path(ids, tokenizer, topology=None):
    collapsed = [int(token) for token, _ in itertools.groupby(ids) if int(token) >= 0]
    return normalize_text(tokenizer.decode(content_ids(collapsed, topology)))


def text_metrics(references, hypotheses):
    assert len(references) == len(hypotheses) and references
    word_errors = sum(Levenshtein.distance(ref.split(), hyp.split()) for ref, hyp in zip(references, hypotheses))
    words = sum(len(ref.split()) for ref in references)
    char_errors = sum(Levenshtein.distance(ref, hyp) for ref, hyp in zip(references, hypotheses))
    chars = sum(len(ref) for ref in references)
    return dict(WER_percent=100 * word_errors / words, CER_percent=100 * char_errors / chars,
                word_errors=word_errors, reference_words=words, char_errors=char_errors,
                reference_characters=chars, utterances=len(references))


def compute_metrics(tokenizer, topology=None):
    def metrics(prediction):
        references = [normalize_text(tokenizer.decode(content_ids(row, topology))) for row in prediction.label_ids]
        hypotheses = [decode_path(row, tokenizer, topology) for row in prediction.predictions]
        return text_metrics(references, hypotheses)
    return metrics


class CheckedTrainer(QuietTrainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite upstream OTTC loss')
        if self.state.global_step < 2:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite gradient: ' + name)
            for head in [model.lm_head, model.lm_weight]:
                assert head.weight.grad is not None and bool(head.weight.grad.abs().sum() > 0)
        if getattr(self, 'smoke_checks', False):
            phase = 'joint' if self.state.global_step < model.frozen_steps_for_regfrozen - 1 else 'frozen'
            with (Path(self.args.output_dir) / 'smoke_checks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.state.global_step, phase=phase, loss=float(loss))) + '\n')
        return loss


def start_tracking(output, config, smoke):
    import wandb
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    saved_path = output / 'wandb_run.json'
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    options = dict(project=os.environ.get('WANDB_PROJECT', defaults['project']),
        entity=os.environ.get('WANDB_ENTITY', defaults['entity']), id=saved.get('id') or wandb.util.generate_id(),
        name='powsm_unigram40k_timit_' + config['recipe'] + ('_smoke' if smoke else ''),
        group=HERE.name, job_type='smoke' if smoke else 'training', config=config, dir=str(output),
        mode=os.environ.get('WANDB_MODE', defaults['mode']),
        settings=wandb.Settings(init_timeout=45, disable_code=True))
    if saved and options['mode'] == 'online':
        options['resume'] = 'allow'
    write_json(saved_path, dict(id=options['id'], project=options['project'], entity=options['entity']))
    if smoke:
        return init_evaluation(**options)
    try:
        return wandb.init(**options)
    except wandb.errors.CommError:
        options['mode'] = 'offline'
        options.pop('resume', None)
        return wandb.init(**options)


class Policy(TrainerCallback):
    def __init__(self, output, config, deadline, run, stop_after_steps=None):
        self.output, self.config, self.deadline, self.run = output, config, deadline, run
        self.stop_after_steps = stop_after_steps
        path = output / 'BEST_DEV.json'
        self.best = json.loads(path.read_text())['WER_percent'] if path.exists() else float('inf')

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        model.step = state.global_step - 1

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() >= self.deadline or (self.stop_after_steps and state.global_step >= self.stop_after_steps):
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        if abs(state.epoch - epoch) > 1e-6:
            return
        path = self.output / 'epoch_weights' / f'epoch-{epoch:03d}.pt'
        metadata = dict(epoch=epoch, step=state.global_step, model_config=model.config.to_dict(),
                        source_sha256=self.config['source_sha256'], tokenizer_sha256=self.config['data']['tokenizer_sha256'])
        if not path.exists():
            atomic_weights(model, path, metadata)
            write_json(path.with_suffix('.json'), dict(**metadata, weights_sha256=sha256(path)))

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        value = metrics.get('eval_WER_percent')
        if value is not None and value < self.best:
            self.best = value
            path = self.output / 'best_dev.pt'
            atomic_weights(model, path, dict(epoch=state.epoch, step=state.global_step))
            write_json(self.output / 'BEST_DEV.json', dict(WER_percent=value, epoch=state.epoch,
                step=state.global_step, checkpoint=str(path), weights_sha256=sha256(path), selection='dev WER only'))

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = dict(logs or {}, optimizer_step=state.global_step, epoch=state.epoch)
        if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in record.values()):
            raise FloatingPointError('Nonfinite training metric')
        with (self.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.run.log(record)


def evaluate(model, rows, processor, tokenizer, destination, microbatch, topology=None):
    model.eval()
    references, hypotheses = [], []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as stream, torch.inference_mode():
        for offset in range(0, len(rows), microbatch):
            subset = rows[offset:offset + microbatch]
            batch = Collator(processor)(subset)
            batch.pop('labels')
            batch = {key: value.cuda() for key, value in batch.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(**batch)
            paths = output.logits.argmax(-1).cpu().tolist()
            lengths = output.hidden_states[1].sum(-1).long().cpu().tolist()
            for row, path, length in zip(subset, paths, lengths):
                ids = path[:length]
                hypothesis = decode_path(ids, tokenizer, topology)
                references.append(row['text'])
                hypotheses.append(hypothesis)
                stream.write(json.dumps(dict(utt=row['utt'], reference=row['text'], hypothesis=hypothesis,
                                             frame_ids=ids), ensure_ascii=True) + '\n')
    result = text_metrics(references, hypotheses)
    write_json(destination.with_suffix('.metrics.json'), result)
    return result


def final_benchmarks(model, rows, processor, tokenizer, output, config, topology=None):
    results = {}
    members = sorted((output / 'epoch_weights').glob('epoch-*.pt'))
    assert len(members) >= 10, 'Missing averaging checkpoints'
    for candidate, count in [('best_dev', 0), ('last5', 5), ('last10', 10)]:
        if not count:
            checkpoint = output / 'best_dev.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['model']
            provenance = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint))
        else:
            selected = members[-count:]
            state, protocol = average(selected, mfa=True)
            provenance = dict(protocol=protocol, members=[dict(path=str(path), **json.loads(path.with_suffix('.json').read_text()))
                                                        for path in selected])
        model.load_state_dict(state, strict=True)
        del state
        results[candidate] = dict(provenance=provenance)
        for split in ['dev', 'test']:
            results[candidate][split] = evaluate(model, rows[split], processor, tokenizer,
                output / 'evaluation' / candidate / (split + '.jsonl'), config['microbatch'], topology)
        write_json(output / 'RESULTS.json', dict(candidates=results, evaluation_uploads=False))
    selected = min(results, key=lambda name: results[name]['dev']['WER_percent'])
    write_json(output / 'RESULTS.json', dict(candidates=results, selected_by_dev_WER=selected,
        evaluation_uploads=False, decoding=config['decoding'],
        normalization=config['data']['normalization'], segmentation='not evaluated: no reference BPE boundaries'))
    write_json(output / 'BENCHMARK_COMPLETE.json', dict(selected_by_dev_WER=selected,
        results_sha256=sha256(output / 'RESULTS.json')))


def check():
    tokenizer = Tokenizer(HERE / 'data/tokenizer')
    token = tokenizer.encode('she')
    assert len(token) == 1
    assert decode_path(token * 3, tokenizer) == 'she'
    assert decode_path(token + [0] + token, tokenizer) == 'she she'
    assert normalize_text(tokenizer.decode(token + token + [-100])) == 'she she'
    assert text_metrics(['she she'], ['she'])['WER_percent'] == 50
    assert text_metrics(['a b'], ['a b'])['CER_percent'] == 0
    topology = build_topology(tokenizer, 'spottc_s3t2')
    states = topology['mapping'][token[0]]
    assert content_ids(states * 2, topology) == token * 2
    assert decode_path(states[:1] + states[1:2] * 4 + states[2:], tokenizer, topology) == 'she'
    assert decode_path(states * 2, tokenizer, topology) == 'she she'
    assert decode_path([states[0], states[2]], tokenizer, topology) == 'she'
    assert content_ids(states + [-100], topology) == token
    processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
        do_normalize=True, return_attention_mask=True)
    collated = ChunkedCollator(Collator(processor), 2)([
        dict(input_values=np.zeros(1600, dtype=np.float32), labels=token),
        dict(input_values=np.ones(800, dtype=np.float32), labels=token + token),
        dict(input_values=np.ones(1200, dtype=np.float32), labels=token)])
    assert collated['logical_batch_size'] == 3 and len(collated['microbatches']) == 2
    assert collated['microbatches'][0]['labels'].tolist() == [token + [-100], token + token]
    expanded = Collator(processor, topology['mapping'])([
        dict(input_values=np.zeros(1600, dtype=np.float32), labels=token * 2),
        dict(input_values=np.ones(800, dtype=np.float32), labels=token)])
    assert expanded['labels'].tolist() == [states * 2, states + [-100] * 3]
    prediction = argparse.Namespace(label_ids=expanded['labels'].numpy(), predictions=expanded['labels'].numpy())
    assert compute_metrics(tokenizer, topology)(prediction)['WER_percent'] == 0
    with tempfile.TemporaryDirectory() as directory:
        model = torch.nn.Linear(2, 2)
        paths = [Path(directory) / f'epoch-{epoch}.pt' for epoch in [1, 2]]
        for epoch, path in enumerate(paths, 1):
            with torch.no_grad():
                model.weight.fill_(epoch)
            atomic_weights(model, path, dict(epoch=epoch, step=epoch * 2))
        averaged, _ = average(paths, mfa=True)
        assert torch.all(averaged['weight'] == 1.5)
        assert torch.load(paths[0], weights_only=False)['step'] == 2
    for split in ['train', 'dev', 'test']:
        rows = [json.loads(line) for line in (HERE / 'data' / (split + '.jsonl')).open()]
        assert all(tokenizer.decode(row['labels']) == row['text'] for row in rows)
        assert all(content_ids([state for token_id in row['labels'] for state in topology['mapping'][token_id]], topology)
                   == row['labels'] for row in rows)
    print('PASS: tokenizer and S3T2 roundtrip, repeated labels, WER/CER, microbatch padding, epoch archive and averaging')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--head', choices=['ottc', 'spottc_s3t2'], default='ottc')
    parser.add_argument('--cost', choices=['jsd', 'ce'], default='jsd')
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--max-seconds', type=int, default=79200)
    args = parser.parse_args()
    if args.check:
        check()
        return
    assert torch.cuda.is_available(), 'Submit on an allocated GPU'
    torch.set_num_threads(4)
    set_seed(42)
    metadata = json.loads((HERE / 'data/metadata.json').read_text())
    sources = source_hashes()
    recipe = args.head + '__' + args.cost
    if not args.smoke:
        gate = json.loads((HERE / 'smoke' / recipe / 'PASS.json').read_text())
        assert gate['source_sha256'] == sources and gate['data'] == metadata, 'Smoke gate or dataset changed'
        assert gate['recipe'] == recipe
    rows = {split: [json.loads(line) for line in (HERE / 'data' / (split + '.jsonl')).open()]
            for split in ['train', 'dev', 'test']}
    for split in rows:
        assert sha256(HERE / 'data' / (split + '.jsonl')) == metadata['manifests_sha256'][split]
    tokenizer = Tokenizer(HERE / 'data/tokenizer')
    assert sha256(HERE / 'data/tokenizer/bpe.model') == metadata['tokenizer_sha256']
    assert sha256(HERE / 'data/tokenizer/tokens.json') == metadata['tokens_sha256']
    topology = build_topology(tokenizer, args.head)
    vocab_size = len(tokenizer.tokens) if topology is None else len(topology['labels'])
    if args.smoke:
        rows['train'] = sorted(rows['train'], key=lambda row: row['duration'], reverse=True)[:12]
        rows['dev'] = rows['dev'][:4]
    config = dict(name=HERE.name, recipe=recipe, head=args.head, data=metadata, source_sha256=sources,
        seed=42, epochs=1 if args.smoke else 40,
        global_batch=4 if args.smoke else 64, microbatch=2, learning_rate=2e-4,
        warmup_steps=0 if args.smoke else 500, weight_decay=.005,
        bins_metric='jsd' if args.cost == 'jsd' else 'crossentropy', forward_type='regfrozen',
        freeze_alignment_after_epochs=30, smoke=args.smoke, evaluation_uploads=False,
        backbone='official Wav2Vec2-large-LV60 self-supervised weights; random output head; no POWSM acoustic weights',
        backbone_sha256=sha256(ROOT / 'reproduction/models/wav2vec2-large-lv60/pytorch_model.bin'),
        output_classes=vocab_size, topology='none' if topology is None else 's3_t2',
        topology_adaptation='none' if topology is None else 'Each nonblank POWSM output ID has 3 ordered states; stable bpeNNNNN surrogate names; no pruning or silence exception',
        topology_training='state_uniform ordered target OTTC; no FSA forward loss or graph-constrained decoding',
        decoding='audio-only argmax collapse; remove blank; existing state-reset decoder for S3T2; native SentencePiece decode; no LM',
        loss_reduction='original summed OTTC loss divided by actual logical batch size',
        checkpoint_policy='atomic model-only weights every completed epoch, retain all; best-dev WER; latest full Trainer state')
    output = HERE / ('smoke' if args.smoke else 'models') / recipe
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = output / 'run_config.json'
    if contract.exists():
        assert json.loads(contract.read_text()) == config, 'Active source or resume contract changed'
    else:
        write_json(contract, config)
        for relative in sources:
            target = output / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    if (output / ('PASS.json' if args.smoke else 'BENCHMARK_COMPLETE.json')).exists():
        return
    if topology is not None:
        write_json(output / 'state_labels.json', topology['labels'])
        write_json(output / 'token_to_states.json', topology['mapping'])
    processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.,
        do_normalize=True, return_attention_mask=True)
    processor.save_pretrained(output / 'processor')
    from ottc.models.sequence.wav2vec2_ottc import wvForOTTC, wvOTTCConfig
    steps_per_epoch = math.ceil(len(rows['train']) / config['global_batch'])
    model = wvForOTTC(wvOTTCConfig(wav2vec2_path=str(ROOT / 'reproduction/models/wav2vec2-large-lv60'),
        vocab_size=vocab_size, hidden_size=1024, bins_metric=config['bins_metric'], forward_type='regfrozen',
        frozen_steps_for_regfrozen=2 if args.smoke else 30 * steps_per_epoch + 1, pad_token_id=0))
    model.freeze_feature_encoder()
    model.wav2vec2_freeze.requires_grad_(False)
    run = start_tracking(output, config, args.smoke)
    training_args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=config['global_batch'],
        per_device_eval_batch_size=config['microbatch'], gradient_accumulation_steps=1,
        num_train_epochs=config['epochs'], learning_rate=config['learning_rate'], warmup_steps=config['warmup_steps'],
        weight_decay=.005, bf16=True, fp16=False, seed=42, data_seed=42, evaluation_strategy='epoch',
        save_strategy='epoch', save_total_limit=2, load_best_model_at_end=False, logging_steps=1 if args.smoke else 10,
        logging_nan_inf_filter=False, report_to=[], dataloader_num_workers=2, dataloader_pin_memory=True,
        remove_unused_columns=False, group_by_length=True, length_column_name='length', max_grad_norm=1.,
        disable_tqdm=True, save_safetensors=True)
    dataset = {split: Dataset.from_list([dict(wav=row['wav'], length=round(row['duration'] * 16000), labels=row['labels'])
                                       for row in rows[split]]) for split in ['train', 'dev']}
    policy = Policy(output, config, time.monotonic() + args.max_seconds, run, args.stop_after_steps)
    trainer = CheckedTrainer(model=model, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['dev'],
        data_collator=ChunkedCollator(Collator(processor, None if topology is None else topology['mapping']), config['microbatch']),
        tokenizer=processor, compute_metrics=compute_metrics(tokenizer, topology), callbacks=[policy])
    trainer.smoke_checks = args.smoke
    resume = get_last_checkpoint(str(output))
    if (output / 'TRAIN_COMPLETE.json').exists():
        assert resume is not None
        trainer._load_from_checkpoint(resume)
        model.step = json.loads((output / 'TRAIN_COMPLETE.json').read_text())['step']
        run.finish()
        assert not args.smoke, 'Interrupted smoke finalization requires inspection'
        final_benchmarks(model, rows, processor, tokenizer, output, config, topology)
        return
    trainer.train(resume_from_checkpoint=resume, ignore_keys_for_eval=['hidden_states', 'attentions'])
    complete = trainer.state.epoch >= config['epochs'] - 1e-5
    write_json(output / 'LATEST_STATE.json', dict(epoch=trainer.state.epoch, step=trainer.state.global_step,
        checkpoint=get_last_checkpoint(str(output)), complete=complete, job=os.environ.get('PBS_JOBID')))
    if not complete:
        run.finish()
        return
    write_json(output / 'TRAIN_COMPLETE.json', dict(epoch=trainer.state.epoch, step=trainer.state.global_step))
    run.finish()
    if args.smoke:
        assert (output / 'epoch_weights/epoch-001.pt').exists() and (output / 'best_dev.pt').exists()
        checkpoint = Path(get_last_checkpoint(str(output)))
        assert (checkpoint / 'optimizer.pt').exists() and (checkpoint / 'scheduler.pt').exists()
        checks = [json.loads(line) for line in (output / 'smoke_checks.jsonl').read_text().splitlines()]
        assert {record['phase'] for record in checks} == {'joint', 'frozen'}
        assert resume is not None and trainer.state.global_step == 3, 'Smoke must exercise an actual checkpoint resume'
        write_json(output / 'PASS.json', dict(source_sha256=sources, data=metadata, recipe=recipe,
            resumed_from=resume, output_classes=vocab_size, steps=trainer.state.global_step,
            peak_cuda_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, phases='joint and frozen alignment', evaluation_uploads=False))
    else:
        final_benchmarks(model, rows, processor, tokenizer, output, config, topology)


if __name__ == '__main__':
    main()