"""Bounded eight-node DDP comparison, one GPU and one rank per node."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup, set_seed

from reproduction.timit_divergences_20261022 import common
from reproduction.timit_divergences_20261022.profile_recipe import gradient_checks
from reproduction.timit_divergences_20261022.train import append, start_wandb

COSTS = ('ag', 'kl', 'jdiv', 'jsd_standard', 'ce', 'kl_reverse')


def merge_counts(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            merge_counts(target[key], value)
        elif isinstance(value, list):
            target[key].extend(value)
        else:
            target[key] += value


def evaluate(model, rows, vocab, processor, output, rank, world):
    helpers = common.scorer()
    counts = helpers['new_counts']()
    collator = common.Collator(processor)
    model.eval()
    subset = rows[rank::world]
    with output.with_name(output.stem + f'.rank{rank}.predictions.jsonl').open('w') as stream, torch.inference_mode():
        for offset in range(0, len(subset), 16):
            batch_rows = subset[offset:offset + 16]
            batch = collator([dict(wav=row['wav'], labels=row['training_target_ids']) for row in batch_rows])
            batch.pop('labels')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(**{key: value.cuda() for key, value in batch.items()})
            for row, path, length in zip(batch_rows, prediction.logits.argmax(-1).cpu().tolist(),
                                         prediction.hidden_states[1].sum(-1).cpu().tolist()):
                path = path[:int(length)]
                hypothesis = common.phones_from_ids(path, vocab, None)
                counts['utterances'] += 1
                counts['refs'].append(common.normalize([phone for phone, _, _ in row['reference']]))
                counts['hyps'].append(common.normalize(hypothesis))
                frame_ids = np.asarray(path)
                frame_ids[frame_ids == 41] = 0
                boundaries = ((np.flatnonzero(frame_ids[1:] != frame_ids[:-1]) + 1) * .01).tolist()
                helpers['score'](counts, row['reference'], boundaries, row['duration'])
                stream.write(json.dumps(dict(utt=row['utt'], hypothesis=hypothesis, frame_token_ids=path)) + '\n')
    gathered = [None] * world if rank == 0 else None
    distributed.gather_object(counts, gathered, dst=0)
    if rank == 0:
        total = helpers['new_counts']()
        for value in gathered:
            merge_counts(total, value)
        result = dict(metrics=helpers['summary'](total), precision='BF16', mode='audio-only greedy and text-free',
                      protocol='existing TIMIT normalization; frozen IF-MDD boundary/R-value scorer; native10ms')
        common.write(output, result)
        return result


def global_batches(lengths, epoch):
    order = np.random.default_rng(42 + epoch).permutation(len(lengths)).tolist()
    batches = []
    for start in range(0, len(order), 1024):
        block = sorted(order[start:start + 1024], key=lambda index: lengths[index], reverse=True)
        batches.extend(block[offset:offset + 256] for offset in range(0, len(block), 256))
    return batches


def save_resume(model, optimizer, scheduler, output, state, config, rank, world):
    local_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                     numpy=np.random.get_state(), python=random.getstate())
    rng_states = [None] * world if rank == 0 else None
    distributed.gather_object(local_rng, rng_states, dst=0)
    if rank == 0:
        target = output / 'latest_resume.pt'
        temporary = target.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), rng_per_rank=rng_states,
                        config=config, **state), temporary)
        temporary.replace(target)
        common.write(output / 'LATEST_STATE.json', dict(**state, checkpoint=str(target)))
    distributed.barrier()


def run_cost(cost, args, rank, world, data, rows, vocab, processor, audit, stop_epoch):
    output = args.output_dir / cost
    config = common.base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    config.update(cost=cost, bins_metric='crossentropy' if cost == 'ce' else cost,
                  name=f'ddp8_{cost}_e{args.epochs}_b256', epochs=args.epochs, global_batch=256,
                  per_rank_batch=32, world_size=world, warmup_steps=125, seed=42,
                  evaluation_uploads=False, checkpoint_policy='every10 epochs; latest full state at budget stop',
                  gradient_checkpointing='non-reentrant for DDP compatibility',
                  source_sha256=common.source_hashes(), data=audit,
                  warmup_policy='125 steps at batch256 matches original 500 steps at batch64 in examples',
                  tail_policy='no duplicated/dropped examples; rank loss multiplied by world/global_count',
                  evaluation_contract='fixed available checkpoint; full dev before test; BF16; no test selection')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        common.snapshot_sources(output, config['source_sha256'])
        common.write(output / 'run_config.json', config)
    distributed.barrier()
    set_seed(42)
    model, topo = common.build_model(config, vocab, math.ceil(len(data['train']) / 256))
    assert topo is None
    model.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.cuda().train()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    ddp = DistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    collator = common.Collator(processor)
    longest = sorted(range(len(rows['train'])), key=lambda index: rows['train'][index]['duration'], reverse=True)[:32]
    typical = np.random.default_rng(42).permutation(len(data['train'])).tolist()[:256]
    profile = []
    for label, indices in [('longest32_per_rank', longest), ('global_batch256', typical[rank::world])]:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        begin = time.time()
        batch = collator([data['train'][int(index)] for index in indices])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = ddp(**{key: value.cuda() for key, value in batch.items()}).loss / len(indices)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = gradient_checks(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        duration = torch.tensor(time.time() - begin, device='cuda')
        distributed.all_reduce(duration, op=distributed.ReduceOp.MAX)
        profile.append(dict(case=label, seconds=duration.item(),
                            peak_GiB=torch.cuda.max_memory_allocated() / 2**30, gradient_norms=gradients))
    if stop_epoch is None:
        estimate = (args.deadline - time.time() - 150) / (len(COSTS) * 19 * profile[-1]['seconds'] * 1.15)
        stop_epoch = min(args.epochs, max(1, int(estimate)))
    shared_stop = torch.tensor(stop_epoch, device='cuda')
    distributed.broadcast(shared_stop, src=0)
    stop_epoch = int(shared_stop.item())
    if rank == 0:
        common.write(output / 'PROFILE_PASS.json', dict(cases=profile, common_stop_epoch=stop_epoch,
                     rank=rank, world_size=world, job=os.environ.get('PBS_JOBID'), host=os.uname().nodename))
        common.write(output / 'TRAINABLE_PARAMETERS.json', common.parameter_manifest(model))
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    del initial, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model.step = -1
    set_seed(42)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, 125, 19 * args.epochs)
    run = start_wandb(output, config, 'training') if rank == 0 else None
    lengths = [row['duration'] for row in rows['train']]
    step = 0
    begin = time.time()
    budget_stop = False
    for epoch in range(stop_epoch):
        model.train()
        batches = global_batches(lengths, epoch)
        for offset, indices in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            batch = collator([data['train'][index] for index in indices[rank::world]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                raw_loss = ddp(**{key: value.cuda() for key, value in batch.items()}).loss
                loss = raw_loss * world / len(indices)
            finite = torch.isfinite(loss).int()
            distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
            assert finite.item(), f'Nonfinite {cost} loss'
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_value = epoch + (offset + 1) / len(batches)
            logged_loss = raw_loss.detach().float()
            distributed.all_reduce(logged_loss)
            if rank == 0 and (step % 5 == 0 or offset + 1 == len(batches)):
                record = dict(epoch=epoch_value, optimizer_step=step, loss=logged_loss.item() / len(indices),
                              learning_rate=scheduler.get_last_lr()[0], gradient_norm=float(gradient_norm),
                              elapsed_seconds=time.time() - begin)
                append(output / 'history.jsonl', record)
                run.log(record)
            stop = torch.tensor(int(rank == 0 and time.time() >= args.deadline - 70), device='cuda')
            distributed.broadcast(stop, src=0)
            if stop.item():
                budget_stop = True
                break
        state = dict(epoch=epoch_value, step=step, planned_epochs=args.epochs,
                     common_stop_epoch=stop_epoch, training_complete=epoch_value >= args.epochs,
                     budget_stop=budget_stop, next_batch=(offset + 1) % len(batches))
        if (epoch + 1) % 10 == 0 and not budget_stop:
            if rank == 0:
                common.atomic_weights(model, output / f'epoch-{epoch + 1:03d}.pt', dict(**state, recipe=config['name']))
            save_resume(model, optimizer, scheduler, output, state, config, rank, world)
        if budget_stop:
            break
    if stop_epoch % 10 or budget_stop:
        save_resume(model, optimizer, scheduler, output, state, config, rank, world)
    results = {split: evaluate(model, rows[split], vocab, processor, output / f'{split}_metrics.json', rank, world)
               for split in ('dev', 'test')}
    if rank == 0:
        common.write(output / ('TRAIN_COMPLETE.json' if state['training_complete'] else 'PAUSED.json'), state)
        common.write(output / 'RESULTS.json', dict(**state, cost=cost, results=results,
                     checkpoint=str(output / 'latest_resume.pt'), test_used_for_selection=False))
        run.summary.update(state)
        run.finish()
    del ddp, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    distributed.barrier()
    return stop_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    distributed.init_process_group('nccl')
    rank, world = distributed.get_rank(), distributed.get_world_size()
    assert world == 8
    if rank == 0:
        assert common.sha(common.SSL / 'pytorch_model.bin') == common.SSL_SHA
        common.write(args.output_dir / 'SSL_VERIFIED.json', dict(path=str(common.SSL), sha256=common.SSL_SHA))
    distributed.barrier()
    config = common.base.recipe('wav2vec2__timit__10ms__mono__jsd_upstream')
    data, rows, vocab, processor, audit = common.load_data(config)
    stop_epoch = None
    for cost in COSTS:
        remaining = torch.tensor(int(rank == 0 and time.time() >= args.deadline - 100), device='cuda')
        distributed.broadcast(remaining, src=0)
        if remaining.item():
            break
        stop_epoch = run_cost(cost, args, rank, world, data, rows, vocab, processor, audit, stop_epoch)
    if rank == 0:
        summaries = {cost: json.loads((args.output_dir / cost / 'RESULTS.json').read_text()) for cost in COSTS
                     if (args.output_dir / cost / 'RESULTS.json').exists()}
        complete = len(summaries) == len(COSTS)
        common.write(args.output_dir / ('SUITE_COMPLETE.json' if complete else 'SUITE_PAUSED.json'),
                     dict(costs=summaries, common_epochs=stop_epoch, planned_epochs=args.epochs,
                          world_size=world, all_costs_evaluated=complete))
    distributed.destroy_process_group()


if __name__ == '__main__':
    main()
