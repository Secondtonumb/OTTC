"""Numerical and end-to-end metric regression tests, runnable on CPU."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(os.environ.get("OTTC_SOURCE", Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("ottc.utils.gamma_decode", ROOT / "ottc/utils/gamma_decode.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
gamma_decode = module.gamma_decode
uniform_transport = module.uniform_transport
pool_gamma_logits = module.pool_gamma_logits


class GammaTests(unittest.TestCase):
    def test_fractional_overlap_and_marginals(self):
        alpha = torch.tensor([.2, .6, .2], dtype=torch.float64)
        gamma = uniform_transport(alpha, 2)
        torch.testing.assert_close(gamma, torch.tensor([[.2, 0], [.3, .3], [0, .2]], dtype=torch.float64))
        torch.testing.assert_close(gamma.sum(1), alpha)
        torch.testing.assert_close(gamma.sum(0), torch.full((2,), .5, dtype=torch.float64))

    def test_logits_logposterior_ce_equivalence(self):
        torch.manual_seed(42)
        logits = torch.randn(19, 7)
        gamma = uniform_transport(torch.rand(19), 8)
        scores = pool_gamma_logits(logits, gamma)
        scores_log = pool_gamma_logits(logits.log_softmax(-1), gamma)
        self.assertEqual(scores.argmax(-1).tolist(), scores_log.argmax(-1).tolist())
        # Direct weighted CE enumeration is the decision rule we implement.
        costs = -(gamma.float().T @ logits.log_softmax(-1))
        self.assertEqual(scores.argmax(-1).tolist(), costs.argmin(-1).tolist())

    def test_transient_frame_insertion_and_duplicate_positions(self):
        logits = torch.tensor([[8., 0.], [0., 9.], [8., 0.]])
        alpha = torch.tensor([.49, .02, .49])
        self.assertEqual(gamma_decode(logits, alpha, 1).tolist(), [0])
        self.assertEqual(gamma_decode(logits, alpha, 2).tolist(), [0, 0])

    def test_probability_average_differs_from_geometric_average(self):
        posterior = torch.tensor([[.99, .01], [.00001, .99999]])
        logits = posterior.log()
        alpha = torch.tensor([.6, .4])
        self.assertEqual(gamma_decode(logits, alpha, 1).tolist(), [1])
        self.assertEqual(gamma_decode(logits, alpha, 1, aggregation='posterior').tolist(), [0])
        with self.assertRaises(ValueError):
            gamma_decode(logits, alpha, 1, aggregation='unknown')

    def test_padding_and_zero_mass_evidence(self):
        logits = torch.tensor([[0., 8.], [float('nan'), float('nan')]])
        self.assertEqual(gamma_decode(logits, torch.tensor([1., 0.]), 1).tolist(), [1])
        self.assertEqual(gamma_decode(logits, torch.tensor([1., float('nan')]), 1,
                                      torch.tensor([True, False])).tolist(), [1])

    def test_precision_scaling_and_empty(self):
        x = torch.tensor([[0., 4.], [0., 5.]], dtype=torch.bfloat16)
        a = torch.tensor([.3, .7])
        self.assertEqual(gamma_decode(x, a, 4).tolist(), [1] * 4)
        self.assertEqual(gamma_decode(x, a * 17, 4).tolist(), [1] * 4)
        self.assertEqual(gamma_decode(x, a, 0).tolist(), [])

    def test_reject_invalid_inputs(self):
        for alpha, count in [(torch.zeros(3), 1), (torch.tensor([-1., 2.]), 1),
                             (torch.tensor([float('nan')]), 1), (torch.ones(2), -1),
                             (torch.ones(2), 1.5)]:
            with self.assertRaises(ValueError):
                uniform_transport(alpha, count)
        with self.assertRaises(ValueError):
            pool_gamma_logits(torch.ones(2, 3), torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            gamma_decode(torch.ones(2, 3), torch.ones(3), 1)

    def test_greedy_length_preserves_blank_splits(self):
        self.assertEqual(module.greedy_token_count([1, 1, 0, 1, 2, 2, 3], {0, 3}), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required by existing OTTC loss")
    def test_matches_existing_bucketized_ottc_ce(self):
        from ottc.ops_loss.ot_loss import batched_ottc_loss_bucketized
        torch.manual_seed(42)
        logits = torch.randn(1, 7, 5, device='cuda')
        alpha = torch.tensor([[1, 3, 2, 4, 2, 3, 1]], device='cuda').float() / 16
        gamma = uniform_transport(alpha[0], 4)
        ids = pool_gamma_logits(logits[0], gamma).argmax(-1)
        targets = torch.nn.functional.one_hot(ids, 5).float()[None]
        beta = torch.full((1, 4), .25, device='cuda')
        loss, *_ = batched_ottc_loss_bucketized(logits, targets, alpha, beta,
                                               torch.ones_like(alpha), torch.ones_like(beta))
        dense = -(gamma.float() * logits[0].log_softmax(-1)[:, ids]).sum()
        torch.testing.assert_close(loss, dense)

    def test_metrics_default_and_reference_length_no_collapse(self):
        spec = importlib.util.spec_from_file_location("gamma_test_metrics", ROOT / "ottc/utils/eval_metrics.py")
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)

        class Tokenizer:
            pad_token_id = 0
            unk_token_id = 3
            def get_vocab(self): return {"<ignored>": 4}
            def convert_tokens_to_ids(self, token): return 4

        class Processor:
            tokenizer = Tokenizer()
            def batch_decode(self, batch, group_tokens=True):
                import itertools
                strings = []
                for ids in batch:
                    if group_tokens: ids = [k for k, _ in itertools.groupby(ids)]
                    strings.append(" ".join(str(i) for i in ids if i not in (0, 3)))
                return strings

        logits = np.array([[[0., 8., 0., 0.], [0., 8., 0., 0.], [99., 0., 0., 0.]]])
        alpha = np.array([[.5, .5, 0.]])
        mask = np.array([[1, 1, 0]])
        labels = np.array([[1, 1, -100]])
        pred = SimpleNamespace(predictions=(logits, (alpha, mask)), label_ids=labels)
        processor = Processor()
        self.assertEqual(metrics.build_compute_metrics(processor, loss_type="ottc")(pred)["wer"], .5)
        self.assertEqual(metrics.build_compute_metrics(processor, loss_type="ottc",
            decoder="gamma_greedy_length")(pred)["wer"], .5)
        self.assertEqual(metrics.build_compute_metrics(processor, loss_type="ottc",
            decoder="gamma_reference_length")(pred)["wer"], 0.)
        np.testing.assert_array_equal(labels, pred.label_ids)
        with self.assertRaises(ValueError):
            metrics.build_compute_metrics(processor, loss_type="ctc", decoder="gamma_greedy_length")


if __name__ == "__main__":
    unittest.main()
