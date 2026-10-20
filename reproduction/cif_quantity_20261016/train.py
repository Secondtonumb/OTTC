"""Paired Buckeye/Libri100 CIF experiments with quantity loss.

Run from the repository root: python -m reproduction.cif_quantity_20261016.train --help
"""
import argparse
from contextlib import nullcontext
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import time

import numpy as np
import torch
from transformers import Wav2Vec2Config

from ottc.models.sequence.wav2vec2_cif import wvCIFConfig, wvForCIF
from reproduction.cif_quantity_20261016.data import ROOT, load_rows, collate, sha, benchmark_rows
from reproduction.cif_quantity_20261016.evaluate import evaluate
from reproduction.segmentation_metrics import SOURCE

HERE = Path(__file__).resolve().parent


class Tracking:
    def __init__(self, output, contract, mode):
        self.output, self.run = output, None
        if mode == 'disabled':
            return
        import wandb
        defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
        identity = output / 'wandb_run.json'
        saved = json.loads(identity.read_text()) if identity.exists() else {}
        project = os.environ.get('WANDB_PROJECT', defaults['project'])
        entity = os.environ.get('WANDB_ENTITY', defaults.get('entity'))
        if saved and (saved['project'], saved['entity']) != (project, entity):
            raise ValueError('Resume must retain the W&B project/entity')
        options = dict(project=project, entity=entity, name=output.name,
                       group='cif_quantity_two_corpora_20261016', config=contract,
                       dir=str(output.resolve()), mode=mode, id=saved.get('id'),
                       settings=wandb.Settings(init_timeout=35))
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
        identity.write_text(json.dumps(dict(id=self.run.id, project=project, entity=entity,
                                            mode=self.run.settings.mode), indent=2))
        self.run.define_metric('step')
        self.run.define_metric('*', step_metric='step')

    def log(self, record):
        with (self.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if self.run is not None:
            self.run.log(record)

    def finish(self, complete, failed=False):
        from reproduction.wandb_policy import log_evaluation_artifact
        if self.run is not None:
            import wandb
            self.run.summary['training_complete'] = complete
            artifact = wandb.Artifact(self.output.name + '-summaries', type='evaluation')
            for path in [self.output / 'run_config.json', self.output / 'history.jsonl',
                         *self.output.glob('final_*.json'), *self.output.glob('benchmark_*.json'), *self.output.glob('*COMPLETE.json')]:
                if path.exists():
                    artifact.add_file(str(path))
            log_evaluation_artifact(self.run, artifact)
            self.run.finish(exit_code=int(failed))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def plan_batches(rows, batch_size, epoch, seed, profile=False):
    if profile:
        # Exercise the actual longest padded batch first; throughput is reported
        # on this deliberately pessimistic batch, never as whole-epoch throughput.
        indices = sorted(range(len(rows)), key=lambda i: rows[i]['duration'], reverse=True)
        return [indices[:batch_size]]
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(len(rows)).tolist()
    batches = []
    for offset in range(0, len(order), batch_size * 16):
        bucket = sorted(order[offset:offset + batch_size * 16], key=lambda i: rows[i]['duration'])
        batches += [bucket[i:i + batch_size] for i in range(0, len(bucket), batch_size)]
    rng.shuffle(batches)
    return batches


def flat_metrics(split, result):
    values = {f'{split}/{k}': result[k] for k in ('raw_PER_percent', 'normalized_nosil_PER_percent',
              'ce_per_phone', 'quantity_per_phone', 'ctc_per_phone', 'ottc_per_utterance', 'quantity_MAE')}
    for mode in ('text_free', 'text_free_frame'):
        for tolerance, metrics in result[mode].items():
            for key in ('precision_pct', 'recall_pct', 'f1_pct', 'r_value_pct'):
                values[f'{split}/{mode}/{key}_{tolerance}ms'] = metrics[key]
    return values


def main():
    defaults = json.loads((ROOT / 'reproduction/wandb_config.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['buckeye','libri100'], required=True)
    parser.add_argument('--ctc-weight', type=float, default=0.0)
    parser.add_argument('--ottc-weight', type=float, default=0.0, help='Replace CTC with original upstream JSD OTTC auxiliary loss')
    parser.add_argument('--quantity-weight', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--encoder', type=Path, default=ROOT / 'reproduction/models/wav2vec2-large-lv60')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.005)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--max-wall-seconds', type=float, default=3000)
    parser.add_argument('--max-steps', type=int, default=0, help='Absolute optimizer step cap; keeps full epoch LR schedule')
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--tiny', action='store_true', help='Random tiny encoder for CPU correctness checks only')
    parser.add_argument('--profile', action='store_true', help='Bounded longest-batch GPU probe; never creates production weights')
    parser.add_argument('--evaluate-on-stop', action='store_true', help='Diagnostic dev/test of a bounded smoke run, marked partial')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default=os.environ.get('WANDB_MODE', defaults['mode']))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_every, args.learning_rate) <= 0:
        parser.error('epochs, batch size, save interval and learning rate must be positive')
    if min(args.max_steps, args.train_limit, args.eval_limit, args.max_wall_seconds, args.warmup_steps) < 0:
        parser.error('limits must be nonnegative')
    if args.profile and not args.max_steps:
        parser.error('--profile requires a finite --max-steps')
    if args.quantity_weight != 1:
        parser.error('This experiment requires quantity loss weight 1')
    if args.ctc_weight and args.ottc_weight:
        parser.error('Choose --ctc-weight or --ottc-weight, not both')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This entry currently supports one process / one GPU')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.tiny:
        parser.error('Use an allocated GPU for the real encoder; CPU checks require --tiny')
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if device.type == 'cuda' else None
    rows, vocab, data_contract = load_rows(args.dataset, args.train_limit, args.eval_limit)
    source_paths = [*HERE.glob('*.py'), ROOT / 'ottc/models/sequence/cif.py',
                    ROOT / 'ottc/models/sequence/wav2vec2_cif.py', SOURCE,
                    ROOT / 'reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py',
                    ROOT / 'reproduction/cif_timit_20261016/evaluate.py']
    contract = dict(experiment='CIF with quantity loss', dataset=args.dataset, **data_contract,
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        encoder=str(args.encoder), tiny=args.tiny, ctc_weight=args.ctc_weight,
        ottc_weight=args.ottc_weight, ottc_metric='original upstream jsd=True',
        ottc_mass='independent linear mass head -> masked softmax; uniform mass per phone occurrence; shared42-ID TIMIT40 inventory, source mapping recorded in data contract',
        quantity_weight=args.quantity_weight, label_smoothing=args.label_smoothing,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, seed=args.seed,
        precision='BF16 autocast, FP32 integration/loss' if dtype else 'FP32',
        profile=args.profile, train_limit=args.train_limit, eval_limit=args.eval_limit,
        split_counts={split: len(value) for split, value in rows.items()},
        target_convention='audited source phone occurrences; no EOS; private BOS input embedding',
        loss_reduction='sum(CE + weighted L1 quantity + weighted CTC) / total target phones + weighted native OTTC sum / utterances',
        selection='maximum dev text_free R-value@20ms; test never used for selection',
        boundary_protocol='frozen IF-MDD directional P/R/F1/R; duration/T; CIF crossings, endpoints excluded',
        decoder='2-layer AR SAN, d=256, heads=4, ffn=2048; tiny uses 1/16/2/32',
        optimizer='AdamW; linear warmup then linear decay; frozen CNN; all transformer layers trainable',
        evaluation_batch_size=8, quantity_loss_disabled=False,
        regfrozen=False, frozen_prior_ottc_models_untouched=True)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = args.output / 'run_config.json'
    checkpoint_path = args.output / 'resume.pt'
    if config_path.exists() and json.loads(config_path.read_text()) != contract:
        raise ValueError('Existing run configuration differs; choose a new output')
    if (args.output / 'TRAIN_COMPLETE.json').exists():
        print('Already complete:', args.output)
        return
    if config_path.exists() and not checkpoint_path.exists():
        raise ValueError('Existing output has no resumable state; use a new output')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
    head_config = dict(vocab_size=len(vocab), ctc_weight=args.ctc_weight,
                       ottc_weight=args.ottc_weight,
                       quantity_weight=args.quantity_weight, label_smoothing=args.label_smoothing)
    if saved:
        model = wvForCIF(wvCIFConfig(**saved['model_config']))
        model.load_state_dict(saved['model'])
    elif args.tiny:
        base = Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=64, conv_dim=(16, 16, 16), conv_stride=(10, 8, 4), conv_kernel=(20, 10, 5),
            num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16, mask_time_prob=0, feat_extract_norm="layer",
            hidden_dropout=0, attention_dropout=0, feat_proj_dropout=0, layerdrop=0)
        model = wvForCIF(wvCIFConfig(encoder_config=base.to_dict(), decoder_dim=16,
            decoder_heads=2, decoder_layers=1, decoder_ffn=32, dropout=0, **head_config))
    else:
        model = wvForCIF.from_encoder_pretrained(args.encoder, **head_config)
    model.freeze_feature_encoder()
    if not args.tiny:
        model.wav2vec2.gradient_checkpointing_enable()
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(rows['train']) / args.batch_size) * args.epochs
    def lr_scale(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        return max(0., (total_steps - step) / max(total_steps - args.warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    progress = dict(epoch=0, cursor=0, step=0, best_dev_r=-math.inf)
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        progress = saved['progress']
        restore_rng(saved['rng'])
        del saved
    config_path.write_text(json.dumps(contract, indent=2) + '\n')
    (args.output / 'vocab_used.json').write_text(json.dumps(vocab, indent=2) + '\n')
    model.config.save_pretrained(args.output)
    runtime = dict(host=platform.node(), pbs_job=os.environ.get('PBS_JOBID'), torch=torch.__version__,
                   cuda=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                   argv=os.sys.argv, smoke=args.tiny or bool(args.train_limit or args.eval_limit), profile=args.profile)
    (args.output / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    rng = rng_state()
    tracking = Tracking(args.output, contract, args.wandb_mode)
    restore_rng(rng)  # telemetry initialization must not change training RNG
    stopped = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *unused: stopped.__setitem__(0, True))
    began = time.monotonic()
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    def save():
        if args.profile:
            return
        atomic_save(dict(model=model.state_dict(), model_config=model.config.to_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), progress=copy.deepcopy(progress),
            rng=rng_state()), checkpoint_path)
    complete, failed = False, True
    try:
        model.train()
        while progress['epoch'] < args.epochs:
            plan = plan_batches(rows['train'], args.batch_size, progress['epoch'], args.seed, args.profile)
            while progress['cursor'] < len(plan) or args.profile:
                if stopped[0] or (args.max_steps and progress['step'] >= args.max_steps) or (
                    args.max_wall_seconds and time.monotonic() - began >= args.max_wall_seconds):
                    stopped[0] = True
                    break
                indices = plan[0] if args.profile else plan[progress['cursor']]
                sync()
                start = time.monotonic()
                batch = collate([rows['train'][i] for i in indices], device)
                sync()
                loaded = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]['lr']
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    out = model(**batch)
                out['loss'].backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not torch.isfinite(out['loss']):
                    raise ValueError('Nonfinite loss')
                sync()
                backward = time.monotonic()
                optimizer.step()
                scheduler.step()
                sync()
                end = time.monotonic()
                progress['step'] += 1
                progress['cursor'] += 1
                denominator = int(out['token_count'])
                record = dict(step=progress['step'], epoch=progress['epoch'] + min(progress['cursor'], len(plan)) / len(plan),
                    train_loss=float(out['loss']), train_ce_sum=float(out['ce_sum']),
                    train_quantity_sum=float(out['quantity_sum']), train_ctc_sum=float(out['ctc_sum']),
                    train_ottc_sum=float(out['ottc_sum']), train_ottc_per_utterance=float(out['ottc_sum']) / len(indices),
                    train_target_phones=denominator, learning_rate=lr, gradient_norm=float(gradient_norm),
                    loading_transfer_seconds=loaded-start, forward_backward_seconds=backward-loaded,
                    optimizer_seconds=end-backward, batch_seconds=end-start,
                    audio_seconds=sum(rows['train'][i]['duration'] for i in indices),
                    utterances=len(indices), max_audio_seconds=max(rows['train'][i]['duration'] for i in indices),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type == 'cuda' else 0)
                record['audio_seconds_per_wall_second'] = record['audio_seconds'] / record['batch_seconds']
                tracking.log(record)
                del out, batch
                if progress['step'] % args.save_every == 0:
                    save()
            if stopped[0]:
                break
            state = rng_state()
            result = evaluate(model, rows['dev'], vocab, args.output / f'dev_epoch_{progress["epoch"]+1:03d}.json', dtype)
            restore_rng(state)
            tracking.log(dict(step=progress['step'], epoch=progress['epoch']+1, **flat_metrics('dev', result)))
            score = result['text_free']['20']['r_value_pct']
            if score > progress['best_dev_r']:
                progress['best_dev_r'] = score
                atomic_save(dict(model=model.state_dict(), model_config=model.config.to_dict(), step=progress['step'],
                                 epoch=progress['epoch']+1, dev_R_value_20ms=score), args.output / 'best_dev_r.pt')
            progress['epoch'] += 1
            progress['cursor'] = 0
            save()
        if args.profile and progress['step'] < args.max_steps:
            raise RuntimeError('Profile stopped before all requested optimizer updates; no PASS marker')
        complete = progress['epoch'] >= args.epochs and not args.profile
        save()
        if complete:
            best = torch.load(args.output / 'best_dev_r.pt', map_location=device, weights_only=False)
            model.load_state_dict(best['model'])
        if complete or args.evaluate_on_stop:
            for split in ('dev', 'test'):
                result = evaluate(model, rows[split], vocab, args.output / f'final_{split}.json', dtype)
                result.update(partial=not complete, tiny=args.tiny, profile=args.profile,
                              checkpoint_selection='best dev R-value@20ms' if complete else 'bounded diagnostic last step',
                              step=best['step'] if complete else progress['step'])
                (args.output / f'final_{split}.json').write_text(json.dumps(result, indent=2) + '\n')
                tracking.log(dict(step=progress['step'], **flat_metrics('final/' + split, result)))
        if complete and not runtime['smoke']:
            for split, cases in benchmark_rows(args.dataset, rows).items():
                result = evaluate(model, cases, vocab, args.output / f'benchmark_{split}.json', dtype)
                result.update(partial=False, checkpoint_selection='best dev R-value@20ms', step=best['step'])
                (args.output / f'benchmark_{split}.json').write_text(json.dumps(result, indent=2) + '\n')
                tracking.log(dict(step=progress['step'], **flat_metrics('benchmark/' + split, result)))
        marker = 'PROBE_COMPLETE.json' if args.profile else ('TRAIN_COMPLETE.json' if complete else 'PAUSED.json')
        (args.output / marker).write_text(json.dumps(dict(progress=progress, runtime=runtime, full_training_complete=complete,
            partial=not complete, wall_seconds=time.monotonic()-began), indent=2) + '\n')
        failed = False
    finally:
        tracking.finish(complete, failed)


if __name__ == '__main__':
    main()
