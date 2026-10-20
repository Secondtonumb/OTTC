"""Atomic recipe checkpoints including private augmentation generators."""
import copy
import os
import random
import uuid
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def world_rank():
    return (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    return copy.deepcopy(value)


def random_state(specaug):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                specaug_seed=specaug.seed,
                specaug_generators={device: gen.get_state() for device, gen in specaug.device_to_generator.items()})


def restore_random(state, specaug):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    specaug.seed = state['specaug_seed']
    specaug.device_to_generator.clear()
    for device, value in state['specaug_generators'].items():
        gen = torch.Generator(device=device)
        gen.set_state(value)
        specaug.device_to_generator[device] = gen


def save(path, model, model_avg, optimizer, scheduler, scaler, specaug, sampler, progress, contract):
    world, rank = world_rank()
    local = dict(random=random_state(specaug), sampler=copy.deepcopy(sampler), scaler=scaler.state_dict())
    ranks = [None]*world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, ranks, dst=0)
    else:
        ranks = [local]
    error = [None]
    if rank == 0:
        try:
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
            state = dict(format_version=2, world_size=world, model=tree_to(model.state_dict(), 'cpu'),
                         model_avg=tree_to(model_avg.state_dict(), 'cpu'), optimizer=tree_to(optimizer.state_dict(), 'cpu'),
                         scheduler=copy.deepcopy(scheduler.state_dict()), rank_states=ranks,
                         progress=copy.deepcopy(progress), contract=copy.deepcopy(contract))
            with temporary.open('wb') as stream:
                torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            error[0] = repr(exc)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError('Checkpoint failed: '+error[0])
    if world > 1:
        dist.barrier()


def load(path, model, model_avg, optimizer, scheduler, scaler, specaug, sampler, contract):
    world, rank = world_rank()
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['world_size'] != world or state['contract'] != contract:
        raise ValueError('Resume world size or recipe contract differs')
    model.load_state_dict(state['model'], strict=True)
    if model_avg is not None:
        model_avg.load_state_dict(state['model_avg'], strict=True)
    saved = state['optimizer']; optimizer.load_state_dict(saved)
    for live_group, old_group in zip(optimizer.param_groups, saved['param_groups']):
        for parameter, index in zip(live_group['params'], old_group['params']):
            if index in saved['state']:
                optimizer.state[parameter] = tree_to(saved['state'][index], parameter.device)
    scheduler.load_state_dict(state['scheduler'])
    # The archived scheduler omits derived fields from its saved state.
    scheduler.past_batches = sum(scheduler.batches_per_epoch[:scheduler.epoch-1])
    scheduler._set_lrs()
    local = state['rank_states'][rank]
    scaler.load_state_dict(local['scaler'])
    sampler.load_state_dict(local['sampler'])
    restore_random(local['random'], specaug)
    optimizer.zero_grad(set_to_none=True)
    return state['progress']
