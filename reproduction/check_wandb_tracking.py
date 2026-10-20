"""CPU-only real Trainer check of live W&B integration; no speech retraining."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reproduction/vendor'), str(ROOT)]
import wandb
import numpy as np
import torch
import transformers
transformers.utils.import_utils._peft_available = False
transformers.utils.import_utils._bitsandbytes_available = False
from transformers import Trainer, TrainingArguments
from ottc.utils.wandb_tracking import start_training_run, finish_training_run, LossCurveCallback


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, input_values, labels):
        logits = self.linear(input_values)
        return dict(loss=((logits-labels)**2).mean(), logits=logits)


def main():
    output = ROOT/'reproduction/results/wandb_integration_check'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        config = dict(continue_if_exist=False, suffix='wandb_cpu_logging_check', dataset='logging-check',
                      loss_type='ctc', wandb_project='ottc-logging-check', wandb_entity='kevingenghaopeng-',
                      wandb_mode='offline')
        run = start_training_run(config, directory)
        captured = []
        original = run.log
        def capture(data, *args, **kwargs):
            captured.append(dict(data))
            return original(data, *args, **kwargs)
        run.log = capture
        wandb.log = capture
        rows = [dict(input_values=torch.tensor([float(i), 1.]), labels=torch.tensor([float(i)])) for i in range(4)]
        args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=3,
                                  per_device_train_batch_size=2, logging_steps=1,
                                  report_to=['wandb'], save_strategy='no', disable_tqdm=True)
        trainer = Trainer(model=Model(), args=args, train_dataset=rows, eval_dataset=rows,
                           compute_metrics=lambda _: dict(wer=.25), callbacks=[LossCurveCallback()])
        try:
            trainer.train()
            trainer.evaluate()
            assert any('train/loss' in r for r in captured)
            assert any('eval/loss' in r for r in captured)
            assert any(r.get('eval/per_percent') == 25 for r in captured)
            ratios = [r['train/loss_relative'] for r in captured if 'train/loss_relative' in r]
            assert len(ratios) == 3 and ratios[0] == 1
            assert json.loads((Path(directory)/'wandb_run.json').read_text())['mode'] == 'offline'
            assert all('train/global_step' in r for r in captured)
            print('PASS: real CPU Trainer logs raw train/eval loss, relative loss, PER%, steps, and offline identity')
        finally:
            finish_training_run()
        (output/'COMPLETE.json').write_text(json.dumps(dict(steps=3, captured_records=len(captured), passed=True)))


if __name__ == '__main__':
    main()
