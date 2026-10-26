"""Focused local tests for full40 contracts, epoch selection and FP32 averaging."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from reproduction.buckeye_full40_seed_sweep_20261025 import common
from reproduction.buckeye_full40_seed_sweep_20261025.evaluate import average, candidate_members


class RecipeTests(unittest.TestCase):
    def test_epoch_checkpoints(self):
        from reproduction.buckeye_full40_seed_sweep_20261025.train import Policy
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = torch.nn.Linear(2, 2)
            model.config = SimpleNamespace(to_dict=lambda: {'identity': 'test'})
            config = dict(name='ce__seed42', seed=42, source_sha256={})
            policy = Policy(output, config, float('inf'), None)
            for epoch, value in ((1, 20.), (2, 25.), (3, 22.), (4, 30.), (5, 19.), (6, 22.)):
                policy = Policy(output, config, float('inf'), None)
                policy.on_evaluate(None, SimpleNamespace(epoch=epoch, global_step=epoch * 285), None,
                                   model=model, metrics={'eval_nosil_PER_percent': value})
                self.assertLessEqual(len(list((output / 'epoch_weights').glob('*.pt'))), 3)
            import json
            self.assertEqual(len(list((output / 'epoch_weights').glob('*.pt'))), 3)
            self.assertEqual(json.loads((output / 'BEST_DEV.json').read_text())['epoch'], 5)
            members = json.loads((output / 'EPOCHS.json').read_text())
            self.assertEqual([item['epoch'] for item in members], [5, 1, 3])
            self.assertTrue(all(Path(item['path']).exists() for item in members))
            for path in (output / 'epoch_weights').glob('*.pt'):
                saved = torch.load(path, weights_only=False)
                self.assertTrue(all(tensor.dtype == torch.float32 for tensor in saved['model'].values()))

    def test_full40_matrix(self):
        rows, vocab, contracts, _ = common.checked_rows()
        self.assertEqual([len(rows[split]) for split in ('train', 'dev', 'test', 'timit_dev', 'timit_test')],
                         [18221, 6417, 6839, 400, 192])
        self.assertEqual(len(vocab), 42)
        configs = [common.recipe(f'{cost}__seed{seed}', contracts) for cost in common.COSTS for seed in common.SEEDS]
        self.assertEqual(len({config['name'] for config in configs}), 12)
        self.assertTrue(all(config['epochs'] == 40 and config['global_batch'] == 64 for config in configs))
        self.assertTrue(all(config['evaluation_candidates'] == ['best_dev', 'best3'] for config in configs))
        self.assertEqual({config['bins_metric'] for config in configs}, {'crossentropy', 'jsd_standard', 'ag', 'jdiv'})

    def test_candidate_selection(self):
        members = [dict(epoch=epoch, dev_PER=abs(epoch - 12)) for epoch in (11, 13, 12)]
        candidates = candidate_members(members)
        self.assertEqual([item['epoch'] for item in candidates['best3']], [12, 11, 13])
        self.assertEqual(set(candidates), {'best_dev', 'best3'})
        self.assertEqual(candidates['best_dev'], [members[2]])
        with self.assertRaises(AssertionError):
            candidate_members(members[:-1])

    def test_fp32_average_and_seed_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(name='ag__seed42', seed=42, source_sha256={})
            members = []
            for epoch in range(1, 4):
                path = root / f'{epoch}.pt'
                torch.save(dict(model={'weight': torch.full((2,), float(epoch)), 'counter': torch.tensor(0)},
                                recipe=config['name'], seed=42, epoch=epoch, step=epoch * 285,
                                model_config={}, source_sha256={}), path)
                members.append(dict(path=str(path), sha256=common.base.sha(path), epoch=epoch, step=epoch * 285))
            target = root / 'average.pt'
            average(members, config, target)
            saved = torch.load(target, weights_only=False)
            self.assertEqual(saved['model']['weight'].dtype, torch.float32)
            torch.testing.assert_close(saved['model']['weight'], torch.full((2,), 2.))
            with self.assertRaises(AssertionError):
                average(members, dict(config, seed=43), root / 'wrong.pt')


if __name__ == '__main__':
    unittest.main()