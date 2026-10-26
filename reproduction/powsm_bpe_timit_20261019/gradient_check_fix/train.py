"""Phase-aware gradient validation without changing the frozen original recipe."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from reproduction.powsm_bpe_timit_20261019 import train as original

torch = original.torch
BASE = original.HERE
BASE_SOURCES = original.source_hashes()
ORIGINAL_CHECK = original.check


def check_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError('Nonfinite gradient: ' + name)
    prediction_grad = model.lm_head.weight.grad
    assert prediction_grad is not None and bool(prediction_grad.abs().sum() > 0), 'Prediction head has no gradient'
    alignment_grad = model.lm_weight.weight.grad
    frozen = model.step >= model.frozen_steps_for_regfrozen - 1
    if frozen:
        assert alignment_grad is None or not bool(alignment_grad.abs().sum() > 0), 'Frozen alignment head received a gradient'
    else:
        assert alignment_grad is not None and bool(alignment_grad.abs().sum() > 0), 'Joint alignment head has no gradient'


class CheckedTrainer(original.QuietTrainer):
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite upstream OTTC loss')
        if self.state.global_step < 2:
            check_gradients(model)
        if getattr(self, 'smoke_checks', False):
            phase = 'joint' if model.step < model.frozen_steps_for_regfrozen - 1 else 'frozen'
            with (Path(self.args.output_dir) / 'smoke_checks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.state.global_step, phase=phase, loss=float(loss))) + '\n')
        return loss


def check():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2)
    model.lm_weight = torch.nn.Linear(2, 1)
    model.frozen_steps_for_regfrozen = 2
    inputs = torch.ones(2, 2)
    model.step = 0
    (model.lm_head(inputs).sum() + model.lm_weight(inputs).sum()).backward()
    check_gradients(model)
    model.zero_grad(set_to_none=True)
    model.step = 1
    with torch.no_grad():
        alignment = model.lm_weight(inputs)
    (model.lm_head(inputs).sum() + alignment.sum()).backward()
    check_gradients(model)
    model.lm_weight.weight.grad = torch.zeros_like(model.lm_weight.weight)
    check_gradients(model)
    for phase, gradient, error in [(0, None, AssertionError), (1, torch.ones_like(model.lm_weight.weight), AssertionError),
                                   (1, torch.full_like(model.lm_weight.weight, float('nan')), FloatingPointError)]:
        model.step = phase
        model.lm_weight.weight.grad = gradient
        try:
            check_gradients(model)
        except error:
            pass
        else:
            raise AssertionError('Invalid gradient was accepted')
    model.step = 1
    model.lm_weight.weight.grad = None
    model.lm_head.weight.grad = None
    try:
        check_gradients(model)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing prediction gradient was accepted')
    ORIGINAL_CHECK()
    print('PASS: joint/frozen gradients, zero/None frozen gradients, missing and nonfinite gradient rejection')


def seed_smoke(args):
    recipe = args.head + '__' + args.cost
    source = BASE / 'smoke' / recipe
    destination = HERE / 'smoke' / recipe
    if not (source / 'LATEST_STATE.json').exists() or (destination / 'LATEST_STATE.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with (source / '.training.lock').open('a') as source_lock, (destination / '.training.lock').open('a') as destination_lock:
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(destination_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((source / 'run_config.json').read_text())
        state = json.loads((source / 'LATEST_STATE.json').read_text())
        assert config['source_sha256'] == BASE_SOURCES and config['recipe'] == recipe
        assert config['smoke'] and state['step'] == 1 and not state['complete'], 'Only the failed step-one smoke can be imported'
        checkpoint = Path(state['checkpoint']).resolve()
        assert checkpoint.parent == source.resolve()
        target = destination / checkpoint.name
        assert not target.exists(), 'Partial migration exists; inspect it before proceeding'
        members = {str(path.relative_to(checkpoint)): original.sha256(path) for path in checkpoint.rglob('*') if path.is_file()}
        temporary = destination / (checkpoint.name + '.importing')
        shutil.copytree(checkpoint, temporary, copy_function=os.link)
        temporary.replace(target)
        shutil.copy2(source / 'smoke_checks.jsonl', destination / 'smoke_checks.jsonl')
        original.write_json(destination / 'RESUME_ORIGIN.json', dict(source=str(checkpoint), source_config_sha256=original.sha256(source / 'run_config.json'),
            checkpoint_files_sha256=members, change='phase-aware gradient assertions only; model, objective, optimizer and RNG state unchanged'))
        original.write_json(destination / 'LATEST_STATE.json', dict(state, checkpoint=str(target)))
        print('Imported existing smoke step 1 without modifying original checkpoint:', target, flush=True)


def main():
    assert json.loads((BASE / 'DISPATCH.json').read_text())['source_sha256'] == BASE_SOURCES, 'Original source contract changed'
    data = HERE / 'data'
    if not data.exists():
        data.symlink_to(BASE / 'data', target_is_directory=True)
    assert data.resolve() == (BASE / 'data').resolve()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--head', default='ottc', choices=['ottc', 'spottc_s3t2'])
    parser.add_argument('--cost', default='jsd', choices=['jsd', 'ce'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check', action='store_true')
    args, _ = parser.parse_known_args()
    if args.smoke and not args.check:
        seed_smoke(args)
    sources = dict(BASE_SOURCES)
    for path in [HERE / 'train.py', BASE / 'scheduling/run_interactive.sh']:
        sources[str(path.relative_to(ROOT))] = original.sha256(path)
    original.HERE = HERE
    original.source_hashes = lambda: sources
    original.CheckedTrainer = CheckedTrainer
    original.check = check
    original.main()


if __name__ == '__main__':
    main()