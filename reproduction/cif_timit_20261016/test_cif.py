"""CPU algorithm, loss, recognition and checkpoint correctness tests."""
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Wav2Vec2Config

from ottc.models.sequence.cif import CIFPhoneHead, CIFMiddleware, integrate_and_fire
from ottc.models.sequence.wav2vec2_cif import wvCIFConfig, wvForCIF
from reproduction.cif_timit_20261016.evaluate import predicted_boundaries


def scalar_reference(features, weights, tail=0.5):
    """Independent sequential integration, including several firings per frame."""
    mass, value, outputs = 0., torch.zeros_like(features[0]), []
    for feature, weight in zip(features, weights):
        left = float(weight)
        while left > 1e-12:
            take = min(left, 1. - mass)
            mass += take
            value = value + take * feature
            left -= take
            if mass >= 1. - 1e-10:
                outputs.append(value)
                mass, value = 0., torch.zeros_like(feature)
    if mass > tail:
        outputs.append(value / mass)
    return torch.stack(outputs) if outputs else features[:0]


class CIFTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def test_paper_example(self):
        features = torch.tensor([[[1.], [2.], [3.], [4.], [5.]]])
        out = integrate_and_fire(features, torch.tensor([[.2, .9, .6, .6, .1]]), torch.ones(1, 5, dtype=torch.bool))
        torch.testing.assert_close(out.embeddings[0, :, 0], torch.tensor([1.8, 3.2]))
        torch.testing.assert_close(out.fire_positions[0], torch.tensor([1 + .8/.9, 3.5]))

    def test_random_sequential_reference_with_multiple_fires(self):
        for scale in (.03, .9, 3.1):
            features = torch.randn(3, 19, 4)
            weights = torch.rand(3, 19) * scale
            mask = torch.arange(19)[None] < torch.tensor([19, 12, 7])[:, None]
            out = integrate_and_fire(features, weights, mask, chunk_size=3)
            for i, length in enumerate(mask.sum(-1).tolist()):
                expected = scalar_reference(features[i, :length], weights[i, :length])
                self.assertEqual(int(out.lengths[i]), len(expected))
                torch.testing.assert_close(out.embeddings[i, :len(expected)], expected, atol=1e-5, rtol=1e-5)

    def test_scaling_quantity_and_multi_fire(self):
        features = torch.tensor([[[2.], [10.]]])
        alpha = torch.tensor([[.2, .8]], requires_grad=True)
        out = integrate_and_fire(features, alpha, torch.ones(1, 2, dtype=torch.bool), torch.tensor([4]))
        torch.testing.assert_close(out.quantity, torch.tensor([1.]))
        torch.testing.assert_close(out.weights.sum(-1), torch.tensor([4.]))
        torch.testing.assert_close(out.embeddings[0, :, 0], torch.tensor([3.6, 10., 10., 10.]))
        self.assertEqual(int(out.lengths[0]), 4)
        out.embeddings.sum().backward()
        self.assertGreater(float(alpha.grad.abs().sum()), 0)

    def test_padding_tail_and_zero_features(self):
        for padded in (False, True):
            length = 4 if padded else 2
            out = integrate_and_fire(torch.zeros(1, length, 3), torch.tensor([[.8, .8] + ([99., 99.] if padded else [])]),
                torch.tensor([[True, True] + ([False, False] if padded else [])]))
            self.assertEqual(int(out.lengths[0]), 2)  # zero-valued fired features are valid
            self.assertTrue(out.valid_mask.all())
            torch.testing.assert_close(out.fire_positions, torch.tensor([[1.25, 2.]]))
        for mass, count in [(0., 0), (.5, 0), (.51, 1), (1., 1), (1.5, 1), (1.51, 2)]:
            out = integrate_and_fire(torch.ones(1, 1, 2), torch.tensor([[mass]]), torch.ones(1, 1, dtype=torch.bool))
            self.assertEqual(int(out.lengths[0]), count)
            self.assertTrue(torch.isfinite(out.embeddings).all())

    def test_integration_gradcheck(self):
        features = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([[.21, .76, .31, .38]], dtype=torch.float64, requires_grad=True)
        mask = torch.ones(1, 4, dtype=torch.bool)
        function = lambda h, a: integrate_and_fire(h, a, mask, torch.tensor([2])).embeddings
        self.assertTrue(torch.autograd.gradcheck(function, (features, weights), atol=1e-5))

    def test_predictor_masks_before_convolution(self):
        module = CIFMiddleware(4, 8, dropout=0).eval()
        features = torch.randn(1, 7, 4)
        mask = torch.tensor([[True, True, True, True, False, False, False]])
        changed = features.clone()
        changed[:, 4:] = 1e4
        torch.testing.assert_close(module(features, mask).quantity, module(changed, mask).quantity)

    def test_optional_ctc_loss_and_encoder_gradients(self):
        for ctc_weight in (0., .5):
            head = CIFPhoneHead(8, 6, decoder_dim=8, decoder_heads=2, decoder_ffn=16,
                                decoder_layers=1, dropout=0, ctc_weight=ctc_weight)
            features = torch.randn(2, 12, 8, requires_grad=True)
            mask = torch.arange(12)[None] < torch.tensor([12, 8])[:, None]
            labels = torch.tensor([[1, 1, 2], [3, 4, -100]])
            if not ctc_weight:
                with patch('torch.nn.functional.ctc_loss', side_effect=AssertionError('CTC must not run')):
                    out = head(features, mask, labels)
                self.assertIsNone(head.ctc_head)
            else:
                out = head(features, mask, labels)
                self.assertGreater(float(out['ctc_sum'].detach()), 0)
            expected = (out['ce_sum'] + out['quantity_sum'] + ctc_weight * out['ctc_sum']) / 5
            torch.testing.assert_close(out['loss'], expected)
            out['loss'].backward()
            for gradient in (features.grad, head.cif.weight_proj.weight.grad):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
            with torch.no_grad():
                torch.testing.assert_close(features.grad[1, 8:], torch.zeros_like(features.grad[1, 8:]))

    def test_ctc_checks_repeated_phone_feasibility(self):
        head = CIFPhoneHead(8, 6, decoder_dim=8, decoder_heads=2, decoder_ffn=16,
                            decoder_layers=1, dropout=0, ctc_weight=.5)
        with self.assertRaisesRegex(ValueError, 'CTC input too short'):
            head(torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.bool), torch.tensor([[1, 1]]))

    def test_free_recognition_keeps_repeated_phone_events(self):
        head = CIFPhoneHead(8, 6, decoder_dim=8, decoder_heads=2, decoder_ffn=16,
                            decoder_layers=1, dropout=0).eval()
        with torch.no_grad():
            head.cif.weight_proj.weight.zero_()
            head.cif.weight_proj.bias.zero_()  # .5 per frame, 3 fires
            head.output_proj.weight.zero_()
            head.output_proj.bias.zero_()
            head.output_proj.bias[2] = 20
        tokens, cif = head.recognize(torch.randn(1, 6, 8), torch.ones(1, 6, dtype=torch.bool))
        self.assertEqual(tokens.tolist(), [[2, 2, 2]])
        boundaries = predicted_boundaries(cif, .12, 6)
        self.assertEqual(len(boundaries['text_free']), 2)
        torch.testing.assert_close(torch.tensor(boundaries['text_free']), torch.tensor([.04, .08]))

    def test_cpu_bf16(self):
        head = CIFPhoneHead(8, 6, decoder_dim=8, decoder_heads=2, decoder_ffn=16,
                            decoder_layers=1, dropout=0, ctc_weight=.5)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = head(torch.randn(1, 12, 8), torch.ones(1, 12, dtype=torch.bool), torch.tensor([[1, 2, 3]]))
        self.assertEqual(out['loss'].dtype, torch.float32)
        out['loss'].backward()
        self.assertTrue(torch.isfinite(out['loss']))

    def test_auxiliary_losses_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'choose CTC or OTTC'):
            CIFPhoneHead(8, 6, ctc_weight=.5, ottc_weight=.5)

    @unittest.skipUnless(torch.cuda.is_available(), 'Original archived OTTC requires CUDA')
    def test_upstream_ottc_value_gradient_and_loss_scale(self):
        from reproduction.zapformer_s3t2_native_20261012.archived_ot_loss import batched_ottc_loss_bucketized
        head = CIFPhoneHead(8, 6, decoder_dim=8, decoder_heads=2, decoder_ffn=16,
                            decoder_layers=1, dropout=0, ottc_weight=.5).cuda()
        features = torch.randn(2, 12, 8, device='cuda', requires_grad=True)
        valid = torch.arange(12, device='cuda')[None] < torch.tensor([12, 8], device='cuda')[:, None]
        labels = torch.tensor([[1, 1, 2], [3, 4, -100]], device='cuda')
        with patch('torch.nn.functional.ctc_loss', side_effect=AssertionError('OTTC must not call CTC')):
            out = head(features, valid, labels)
        self.assertIsNone(head.ctc_head)
        target_mask = labels != -100
        logits = head.ottc_head(features)
        mass = head.ottc_mass_head(features).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(-1)
        targets = torch.nn.functional.one_hot(labels.masked_fill(~target_mask, 0), 6).float()
        beta = target_mask.float() / target_mask.sum(-1, keepdim=True)
        direct = batched_ottc_loss_bucketized(logits, targets, mass, beta,
                                              amask=valid, bmask=target_mask, jsd=True)[0]
        torch.testing.assert_close(out['ottc_sum'], direct)
        actual_gradient = torch.autograd.grad(out['ottc_sum'], features, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(direct, features, retain_graph=True)[0]
        torch.testing.assert_close(actual_gradient, reference_gradient)
        torch.testing.assert_close(out['loss'], (out['ce_sum'] + out['quantity_sum']) / 5 + .5 * direct / 2)
        out['loss'].backward()
        for gradient in (features.grad, head.ottc_head.weight.grad, head.ottc_mass_head.weight.grad,
                         head.cif.weight_proj.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_save_load_without_original_encoder_path(self):
        base = Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, conv_dim=(8, 8, 8), conv_stride=(5, 2, 2), conv_kernel=(10, 3, 3),
            num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=8, mask_time_prob=0)
        net = wvForCIF(wvCIFConfig(encoder_config=base.to_dict(), vocab_size=6, decoder_dim=8,
            decoder_heads=2, decoder_layers=1, decoder_ffn=16, dropout=0, ctc_weight=.5)).eval()
        values, mask = torch.randn(1, 200), torch.ones(1, 200, dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            net.save_pretrained(directory)
            restored, info = wvForCIF.from_pretrained(directory, output_loading_info=True)
            self.assertFalse(info['missing_keys'] or info['unexpected_keys'])
            restored.eval()
            with torch.no_grad():
                actual = net.recognize(values, mask)
                expected = restored.recognize(values, mask)
            torch.testing.assert_close(actual['tokens'], expected['tokens'])
            torch.testing.assert_close(actual['cif'].fire_positions, expected['cif'].fire_positions)


if __name__ == '__main__':
    unittest.main()
