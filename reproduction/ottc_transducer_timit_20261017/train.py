"""Blank-free OTTC transducer on the frozen TIMIT40 phone-occurrence data.

python -m reproduction.ottc_transducer_timit_20261017.train --help
"""
import argparse
import copy
import fcntl
import inspect
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
import speechbrain
import transformers
from transformers import Wav2Vec2Config

from ottc.models.sequence.wav2vec2_ottc_transducer import wvOTTCTransducerConfig, wvForOTTCTransducer
from reproduction.cif_timit_20261016.data import ROOT, load_rows, collate, sha
from reproduction.segmentation_metrics import SOURCE
from .common import Tracking, atomic_save, write_json, identity, rng_state, restore_rng
from .evaluate import evaluate, flat_metrics

HERE = Path(__file__).resolve().parent


def plan_batches(rows, size, epoch, seed):
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(len(rows)).tolist()
    batches = []
    for start in range(0, len(order), size*16):
        bucket = sorted(order[start:start+size*16], key=lambda i: rows[i]['duration'])
        batches += [bucket[j:j+size] for j in range(0, len(bucket), size)]
    rng.shuffle(batches)
    return batches


def profile_batch(rows, size, step, seed):
    if step % 4 == 0:
        return sorted(range(len(rows)), key=lambda i: rows[i]['duration'], reverse=True)[:size], 'longest_audio'
    if step % 4 == 1:
        return sorted(range(len(rows)), key=lambda i: len(rows[i]['targets']['ottc_arpabet']), reverse=True)[:size], 'most_target_phones'
    if step % 4 == 3:
        return sorted(range(len(rows)), key=lambda i: rows[i]['duration'])[:size], 'shortest_audio'
    return np.random.default_rng(seed).permutation(len(rows))[:size].tolist(), 'representative'


def tiny_config():
    return Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, conv_dim=(16,16,16), conv_stride=(10,8,4), conv_kernel=(20,10,5),
        num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16, mask_time_prob=0,
        hidden_dropout=0, attention_dropout=0, feat_proj_dropout=0, layerdrop=0).to_dict()


def average_checkpoints(paths, contract_id):
    if not paths:
        raise ValueError('cannot average an empty checkpoint set')
    averaged, metadata, model_config = None, [], None
    for path in paths:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['contract_id'] != contract_id:
            raise ValueError('checkpoint identity mismatch')
        if model_config is not None and saved['model_config'] != model_config:
            raise ValueError('model configuration mismatch')
        model_config = saved['model_config']
        metadata.append(dict(path=str(path.resolve()), epoch=saved['epoch'], step=saved['step'], sha256=sha(path)))
        if averaged is None:
            averaged = {k: v.double() if v.is_floating_point() else v.clone() for k,v in saved['model'].items()}
        else:
            if averaged.keys() != saved['model'].keys():
                raise ValueError('checkpoint state keys differ')
            for key, value in saved['model'].items():
                if value.is_floating_point():
                    averaged[key].add_(value.double())
                elif not torch.equal(averaged[key], value):
                    raise ValueError('nonfloating model buffers differ')
    for key, value in averaged.items():
        if value.is_floating_point():
            averaged[key] = (value/len(paths)).to(saved['model'][key].dtype)
    return dict(model=averaged, model_config=model_config, members=metadata, contract_id=contract_id)


def final_benchmarks(model, rows, vocab, output, dtype, contract_id, epoch_count, tracking, step):
    # Freeze all candidate definitions and score development before any test.
    runtime = json.loads((output/'runtime.json').read_text())
    candidates = dict(best_dev=output/'best_dev.pt')
    for count in (5, 10):
        if epoch_count >= count:
            path = output/f'final_{count}_average.pt'
            if not path.exists():
                weights = average_checkpoints([output/f'epoch_{e:03d}.pt'
                    for e in range(epoch_count-count+1, epoch_count+1)], contract_id)
                atomic_save(weights, path)
                write_json(output/f'final_{count}_members.json', dict(members=weights['members'], contract_id=contract_id))
                del weights
            candidates[f'final_{count}'] = path
    dev_results = {}
    for name, path in candidates.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['contract_id'] != contract_id:
            raise ValueError('benchmark checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        del saved
        result = evaluate(model, rows['dev'], vocab, output/f'benchmark_{name}_dev.json', dtype)
        result.update(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path), contract_id=contract_id,
            partial=runtime['partial'], tiny=runtime['tiny'])
        write_json(output/f'benchmark_{name}_dev.json', result)
        dev_results[name] = result
        tracking.log(dict(step=step, **flat_metrics(f'benchmark/{name}/dev', result)))
    selected = min(dev_results, key=lambda name: dev_results[name]['raw_PER_percent'])
    write_json(output/'DEV_SELECTION.json', dict(selected=selected, rule='minimum dev raw PER; tie prefers best-dev then final-5',
        dev_PER={k:v['raw_PER_percent'] for k,v in dev_results.items()}, test_used=False))
    for name, path in candidates.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['contract_id'] != contract_id:
            raise ValueError('benchmark checkpoint identity mismatch')
        model.load_state_dict(saved['model'])
        del saved
        result = evaluate(model, rows['test'], vocab, output/f'benchmark_{name}_test.json', dtype)
        result.update(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path), selected_using_dev=name==selected,
            contract_id=contract_id, test_used_for_selection=False, partial=runtime['partial'], tiny=runtime['tiny'])
        write_json(output/f'benchmark_{name}_test.json', result)
        tracking.log(dict(step=step, **flat_metrics(f'benchmark/{name}/test', result)))


def main():
    defaults = json.loads((ROOT/'reproduction/wandb_config.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--encoder', type=Path, default=ROOT/'reproduction/models/wav2vec2-large-lv60')
    p.add_argument('--length-weight', type=float, default=1.)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8, help='microbatch size; profile before increasing')
    p.add_argument('--accumulation-steps', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.005)
    p.add_argument('--warmup-steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-output-tokens', type=int, default=512)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--max-steps', type=int, default=0, help='absolute optimizer update cap; resumable')
    p.add_argument('--max-wall-seconds', type=float, default=3000)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--eval-limit', type=int, default=0)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--evaluate-on-stop', action='store_true')
    p.add_argument('--wandb-mode', choices=['online','offline','disabled'], default=os.environ.get('WANDB_MODE', defaults['mode']))
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.save_every, args.max_output_tokens) <= 0:
        p.error('epochs, batch sizes, save interval and decoding cap must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not all(math.isfinite(v) and v >= 0
            for v in (args.length_weight, args.weight_decay, args.max_wall_seconds)):
        p.error('invalid learning rate, length weight, decay or wall limit')
    if args.length_weight <= 0:
        p.error('positive length weight is required for text-free decoding')
    if min(args.max_steps, args.train_limit, args.eval_limit, args.warmup_steps) < 0:
        p.error('limits must be nonnegative')
    if args.profile and args.max_steps < 4:
        p.error('profile requires --max-steps >= 4 (longest audio, most phones, representative, shortest audio)')
    if args.profile and (not args.evaluate_on_stop or args.eval_limit < 1 or args.train_limit):
        p.error('profile requires full training inventory, --evaluate-on-stop, and --eval-limit >= 1')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('one process / one GPU recipe')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.tiny:
        p.error('real encoder and original OTTC require an allocated CUDA GPU')
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if device.type == 'cuda' else None
    rows, vocab, data_contract = load_rows(args.train_limit, 0 if args.profile else args.eval_limit)
    if args.profile:
        for split in ('dev', 'test'):
            # Exercise the longest held-out inputs as well as long training audio.
            rows[split] = sorted(rows[split], key=lambda r:r['duration'], reverse=True)[:args.eval_limit or 8]
    source_paths = [HERE/'train.py', HERE/'evaluate.py', HERE/'common.py', ROOT/'ottc/models/sequence/ottc_transducer.py',
        ROOT/'ottc/models/sequence/wav2vec2_ottc_transducer.py', SOURCE,
        ROOT/'reproduction/cif_timit_20261016/data.py', ROOT/'reproduction/cif_timit_20261016/evaluate.py',
        ROOT/'reproduction/segmentation_metrics.py',
        ROOT/'reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py']
    from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    external = {name:sha(Path(inspect.getfile(obj))) for name,obj in dict(
        speechbrain_joint=Transducer_joint).items()}
    effective = args.batch_size*args.accumulation_steps
    contract = dict(experiment='TIMIT40 blank-free autoregressive joint-OTTC transducer', **data_contract,
        sources_sha256={str(path.relative_to(ROOT)):sha(path) for path in source_paths},
        external_sources_sha256=external, torch=torch.__version__, transformers=transformers.__version__,
        speechbrain=speechbrain.__version__, encoder=str(args.encoder.resolve()), tiny=args.tiny,
        encoder_config_sha256=None if args.tiny else sha(args.encoder/'config.json'),
        encoder_weights_sha256=None if args.tiny else sha(args.encoder/'pytorch_model.bin'),
        length_weight=args.length_weight, ottc_metric='original frozen upstream jsd=True; history-conditioned Joint costs on original CDF transport pairs',
        inventory='external frozen42 IDs unchanged; internal40 phone outputs map to external1..40; no blank/pad/unk/EOS output; input-only BOS40',
        count='log count = log(valid_frames) + Linear(masked_mean_encoder_state); Huber to log(reference_count); inference round-half-up with fixed safety cap',
        joint='SpeechBrain Transducer_joint(sum,Tanh) + Linear40; sparse positive transport pairs; no RNNT loss/container',
        predictor='2-layer unidirectional LSTM 256, embedding256, joint256; tiny1/16/16/16',
        epochs=args.epochs, microbatch=args.batch_size, accumulation_steps=args.accumulation_steps, effective_batch=effective,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, seed=args.seed,
        precision='BF16 encoder/joint, FP32 mass/count/original upstream JSD' if dtype else 'FP32',
        loss_reduction='(original upstream joint-OTTC sum + length_weight * log-count Huber sum) / utterances in entire optimizer batch',
        optimizer='AdamW; frozen CNN, trainable transformer; linear warmup/decay; gradient clipping1',
        checkpoint_policy='atomic weights every complete epoch, retain all; best dev raw PER; full latest optimizer/scheduler/RNG/cursor; final5/final10 dev-only selection',
        selection='minimum dev raw PER; free mass-quantile P/R/F1/R and count MAE also reported',
        max_output_tokens=args.max_output_tokens, profile=args.profile,
        train_limit=args.train_limit, eval_limit=args.eval_limit, split_counts={k:len(v) for k,v in rows.items()})
    contract_id = identity(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path, checkpoint_path = args.output/'run_config.json', args.output/'resume.pt'
    if config_path.exists() and json.loads(config_path.read_text()) != contract:
        raise ValueError('configuration/source/data identity changed; use a new output directory')
    if (args.output/'TRAIN_COMPLETE.json').exists() or (args.output/'SMOKE_COMPLETE.json').exists():
        print('Already complete:', args.output)
        return
    if config_path.exists() and not checkpoint_path.exists():
        raise ValueError('existing output has no resumable state; choose a new output')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
    head = dict(phone_ids=list(range(1,41)), external_vocab_size=len(vocab), length_weight=args.length_weight, max_output_tokens=args.max_output_tokens)
    if saved:
        if saved['contract_id'] != contract_id:
            raise ValueError('resume identity mismatch')
        model = wvForOTTCTransducer(wvOTTCTransducerConfig(**saved['model_config']))
        model.load_state_dict(saved['model'])
    elif args.tiny:
        model = wvForOTTCTransducer(wvOTTCTransducerConfig(encoder_config=tiny_config(), embedding_dim=16,
            predictor_dim=16, predictor_layers=1, joint_dim=16, dropout=0, **head))
    else:
        model = wvForOTTCTransducer.from_encoder_pretrained(args.encoder, **head)
    model.freeze_feature_encoder()
    if not args.tiny:
        model.wav2vec2.gradient_checkpointing_enable()
    model.to(device)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(rows['train'])/effective)*args.epochs
    def lr_scale(step):
        if step < args.warmup_steps:
            return (step+1)/max(1, args.warmup_steps)
        return max(0., (total_steps-step)/max(1, total_steps-args.warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    progress = dict(epoch=0, cursor=0, step=0, best_dev_per=None)
    if saved:
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        progress = saved['progress']; restore_rng(saved['rng']); del saved
    write_json(config_path, contract)
    write_json(args.output/'vocab_used.json', vocab)
    runtime = dict(host=platform.node(), job=os.environ.get('PBS_JOBID'), device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        partial=args.tiny or bool(args.train_limit or args.eval_limit), profile=args.profile, tiny=args.tiny)
    write_json(args.output/'runtime.json', runtime)
    rng = rng_state(); tracking = Tracking(args.output, contract, args.wandb_mode); restore_rng(rng)
    stopped = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *unused: stopped.__setitem__(0, True))
    began, complete, failed = time.monotonic(), False, True
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    def save():
        if not args.profile:
            atomic_save(dict(model=model.state_dict(), model_config=model.config.to_dict(), contract_id=contract_id,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), progress=copy.deepcopy(progress), rng=rng_state()), checkpoint_path)
    def weights(epoch):
        return dict(model=model.state_dict(), model_config=model.config.to_dict(), contract_id=contract_id,
            epoch=epoch, step=progress['step'])
    try:
        save()  # An interrupted first update still has a valid starting state.
        model.train()
        while progress['epoch'] < args.epochs:
            plan = plan_batches(rows['train'], effective, progress['epoch'], args.seed)
            while progress['cursor'] < len(plan) or args.profile:
                if stopped[0] or (args.max_steps and progress['step'] >= args.max_steps) or (
                        args.max_wall_seconds and time.monotonic()-began >= args.max_wall_seconds):
                    stopped[0] = True
                    break
                indices, case = profile_batch(rows['train'], effective, progress['step'], args.seed) if args.profile else (plan[progress['cursor']], 'training')
                phones = sum(len(rows['train'][i]['targets']['ottc_arpabet']) for i in indices)
                optimizer.zero_grad(set_to_none=True)
                lr, load_time, compute_time, length_total, ottc_total = optimizer.param_groups[0]['lr'], 0., 0., 0., 0.
                sync(); start = time.monotonic()
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
                for offset in range(0, len(indices), args.batch_size):
                    subset = indices[offset:offset+args.batch_size]
                    tick = time.monotonic()
                    batch = collate([rows['train'][i] for i in subset], device)
                    sync(); loaded = time.monotonic(); load_time += loaded-tick
                    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                        out = model(**batch)
                        loss = (out['ottc_sum']+args.length_weight*out['length_sum'])/len(indices)
                    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                    loss.backward()
                    sync(); compute_time += time.monotonic()-loaded
                    length_total += float(out['length_sum'].detach()); ottc_total += float(out['ottc_sum'].detach())
                    del out, loss, batch
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                tick = time.monotonic(); optimizer.step(); scheduler.step(); sync(); end = time.monotonic()
                progress['step'] += 1; progress['cursor'] += 1
                audio_seconds = sum(rows['train'][i]['duration'] for i in indices)
                tracking.log(dict(step=progress['step'], epoch=progress['epoch']+min(progress['cursor'],len(plan))/len(plan),
                    train_loss=(ottc_total+args.length_weight*length_total)/len(indices), train_length_sum=length_total,
                    train_length_log_huber_per_utterance=length_total/len(indices), train_ottc_sum=ottc_total, train_ottc_per_utterance=ottc_total/len(indices),
                    learning_rate=lr, gradient_norm=float(gradient), target_phones=phones, utterances=len(indices),
                    loading_transfer_seconds=load_time, forward_backward_seconds=compute_time, optimizer_seconds=end-tick,
                    batch_seconds=end-start, audio_seconds=audio_seconds, audio_seconds_per_wall_second=audio_seconds/(end-start),
                    max_audio_seconds=max(rows['train'][i]['duration'] for i in indices), profile_case=case,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else 0))
                if progress['step'] % args.save_every == 0: save()
            if stopped[0]: break
            epoch = progress['epoch']+1
            snapshot = args.output/f'epoch_{epoch:03d}.pt'
            atomic_save(weights(epoch), snapshot)
            manifest_path = args.output/'checkpoint_manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            manifest[str(epoch)] = dict(path=str(snapshot.resolve()), epoch=epoch, step=progress['step'],
                sha256=sha(snapshot), contract_id=contract_id, model_config_id=identity(model.config.to_dict()))
            write_json(manifest_path, manifest)
            rng = rng_state()
            result = evaluate(model, rows['dev'], vocab, args.output/f'dev_epoch_{epoch:03d}.json', dtype)
            restore_rng(rng)
            tracking.log(dict(step=progress['step'], epoch=epoch, **flat_metrics('dev', result)))
            if progress['best_dev_per'] is None or result['raw_PER_percent'] < progress['best_dev_per']:
                progress['best_dev_per'] = result['raw_PER_percent']
                atomic_save(dict(**weights(epoch), dev_PER=progress['best_dev_per']), args.output/'best_dev.pt')
            progress.update(epoch=epoch, cursor=0)
            save()
        complete = progress['epoch'] >= args.epochs and not args.profile
        save()
        if complete:
            final_benchmarks(model, rows, vocab, args.output, dtype, contract_id, args.epochs, tracking, progress['step'])
        elif args.evaluate_on_stop:
            for split in ('dev','test'):
                result = evaluate(model, rows[split], vocab, args.output/f'diagnostic_{split}.json', dtype)
                result.update(partial=True, tiny=args.tiny, profile=args.profile, checkpoint_selection='bounded last update')
                write_json(args.output/f'diagnostic_{split}.json', result)
                tracking.log(dict(step=progress['step'], **flat_metrics('diagnostic/'+split, result)))
        if args.profile and (progress['step'] < args.max_steps or not args.evaluate_on_stop):
            raise RuntimeError('profile must finish all updates and held-out diagnostics before PASS')
        production_complete = complete and not runtime['partial']
        marker = 'PROFILE_PASS.json' if args.profile else ('TRAIN_COMPLETE.json' if production_complete else 'SMOKE_COMPLETE.json' if complete else 'PAUSED.json')
        write_json(args.output/marker, dict(progress=progress, runtime=runtime, contract_id=contract_id,
            full_training_complete=production_complete, wall_seconds=time.monotonic()-began))
        failed = False
    finally:
        tracking.finish(complete=complete and not runtime['partial'], failed=failed)


if __name__ == '__main__':
    main()
