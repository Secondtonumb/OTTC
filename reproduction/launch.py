"""Launch the official entry point with pinned dependencies and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'reproduction/vendor'), str(ROOT)]
import torch
import transformers
import datasets
# These optional packages in the shared environment are unused by this recipe.
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
import run_exp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', choices=['ottc', 'ctc', 'softdtw', 'softdtweuc', 'softdtw_1', 'softdtw_0.01'], required=True)
    parser.add_argument('--bins-metric', choices=['crossentropy', 'jsd', 'jsd_standard', 'euclidian'], default='crossentropy')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    # Honor explicit offline/disabled mode; old recipes set WANDB_DISABLED.
    os.environ.pop('WANDB_DISABLED', None)
    os.environ.setdefault('WANDB_LOG_MODEL', 'false')
    os.environ.setdefault('WANDB_WATCH', 'false')
    suffix = ('smoke_' if args.smoke else 'runexp_') + args.loss + '_fuse_phoneme'
    if args.bins_metric != 'crossentropy':
        if args.loss != 'ottc':
            parser.error('--bins-metric applies only to OTTC')
        suffix += '_' + args.bins_metric
    argv = ['run_exp.py', '--dataset', 'timit_fused_phoneme', '--train_name', 'train',
            '--test_name', 'kaldi_test', '--loss_type', args.loss,
            '--prepare_dataset', 'prepare_dataset_for_timit_phoneme',
            '--batch_size', '64', '--learning_rate', '2e-4' if args.loss == 'ottc' else '5e-5',
            '--num_train_epochs', '40', '--suffix', suffix]
    if args.loss == 'ottc':
        argv += ['--forward_type', 'regfrozen', '--frozen_epochs_for_regfrozen', '30',
                 '--bins_metric', args.bins_metric]
    else:
        argv += ['--force_lowercase', 'True']
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'Run on allocated GPU'
    provenance = dict(command=argv, smoke=args.smoke, seed=42, bins_metric=args.bins_metric,
                      softdtw=dict(gamma={'softdtw': .1, 'softdtweuc': .1, 'softdtw_1': 1., 'softdtw_0.01': .01}[args.loss],
                                   cost='euc' if args.loss == 'softdtweuc' else 'ce', normalize=False)
                        if args.loss.startswith('softdtw') else None,
                      job=os.environ.get('PBS_JOBID'),
                      hostname=platform.node(), gpu=torch.cuda.get_device_name(),
                      python=sys.version, torch=torch.__version__,
                      transformers=transformers.__version__, datasets=datasets.__version__,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), ROOT/'run_exp.py', *sorted((ROOT/'ottc').rglob('*.py'))]})
    output = ROOT/'reproduction/results'/suffix
    output.mkdir(parents=True, exist_ok=True)
    (output/'runtime.json').write_text(json.dumps(provenance, indent=2))
    print('RUNTIME', json.dumps(provenance), flush=True)
    if args.smoke:
        original = run_exp.create_trainer
        def smoke_trainer(model, dataset, processor, config, metrics_fn, model_output_dir):
            short = datasets.DatasetDict({config['train_name']: dataset[config['train_name']].select(range(128)),
                                          config['test_name']: dataset[config['test_name']].select(range(16))})
            trainer = original(model, short, processor, config, metrics_fn, model_output_dir)
            trainer.args.max_steps = 2
            return trainer
        run_exp.create_trainer = smoke_trainer
    sys.argv = argv
    run_exp.main()
    (output/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(job=provenance['job'], smoke=args.smoke)))


if __name__ == '__main__':
    main()
