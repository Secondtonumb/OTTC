import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from reproduction.buckeye_full40_retry48_20261026 import runtime
from reproduction.libri100_timit_seed_sweep_20261022.train import QuietTrainer, ChunkedCollator


class RuntimeTests(unittest.TestCase):
    def test_split_gradients_and_tail_keep_original_mean(self):
        # Check the actual inherited training_step, including its loss reduction.
        for size in (64, 45):
            x = torch.arange(size, dtype=torch.float32).reshape(-1, 1) / size
            reference = torch.nn.Linear(1, 1, bias=False)
            split = torch.nn.Linear(1, 1, bias=False)
            split.load_state_dict(reference.state_dict())
            (reference(x).square().sum() / size).backward()

            class Model(torch.nn.Module):
                def forward(self, input_values):
                    return SimpleNamespace(loss=split(input_values).square().sum())

            from contextlib import nullcontext
            stub = SimpleNamespace(_prepare_inputs=lambda batch: batch,
                compute_loss_context_manager=nullcontext,
                accelerator=SimpleNamespace(backward=lambda loss: loss.backward()))
            collator = ChunkedCollator(lambda rows: {'input_values': torch.stack(rows)}, 48)
            inputs = collator(list(x))
            self.assertEqual([len(b['input_values']) for b in inputs['microbatches']],
                             [48, 16] if size == 64 else [45])
            QuietTrainer.training_step(stub, Model(), inputs)
            torch.testing.assert_close(split.weight.grad, reference.weight.grad)

    def test_probe_forces_save_and_evaluation_after_resume(self):
        control = SimpleNamespace(should_save=False, should_evaluate=False)
        runtime.ProbeCallback().on_step_end(None, None, control)
        self.assertTrue(control.should_save and control.should_evaluate)

    def test_runtime_preserves_optimizer_batch(self):
        with patch.object(runtime.train, 'TrainingArguments') as args, \
             patch.object(runtime.train, 'ChunkedCollator') as collator:
            runtime.patch_runtime()
            runtime.train.TrainingArguments(per_device_train_batch_size=64, per_device_eval_batch_size=64)
            self.assertEqual(args.call_args.kwargs['per_device_train_batch_size'], 64)
            self.assertEqual(args.call_args.kwargs['per_device_eval_batch_size'], 48)
            runtime.train.ChunkedCollator(None, 64)
            collator.assert_called_once_with(None, 48)


if __name__ == '__main__':
    unittest.main()
