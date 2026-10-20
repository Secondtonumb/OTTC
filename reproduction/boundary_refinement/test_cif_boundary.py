"""CPU tests: python -m unittest discover -s reproduction/boundary_refinement."""

import io
import unittest

import torch

from ottc.models.alignment import CIFBoundaryRefiner, boundary_loss, mass_quantiles
from ottc.models.alignment import refine_from_ottc_mass


class FrozenMassTest(unittest.TestCase):
    def test_accept_fractional_candidate_without_parameters(self):
        out = refine_from_ottc_mass(torch.tensor([0.4, 0.6]), torch.tensor([0.1]), 0.2)
        torch.testing.assert_close(
            out.times, torch.tensor([0.1 + 0.1 * (0.5 - 0.4) / 0.6])
        )
        self.assertTrue(out.accepted.item())

    def test_distant_crossing_falls_back_without_pasting_fraction(self):
        out = refine_from_ottc_mass(torch.tensor([0.4, 0.6]), torch.tensor([0.05]), 0.2)
        torch.testing.assert_close(out.times, torch.tensor([0.05]))
        self.assertFalse(out.accepted.item())
        self.assertGreater(out.global_times.item(), 0.1)

    def test_same_as_existing_native_ot(self):
        # Independent scalar implementation mirrors native_ot_spans.
        mass = [0.2, 0.9, 0.6, 0.6]
        n, duration = 6, 0.08
        expected = []
        for j in range(1, n):
            threshold = sum(mass) * j / n
            before = 0.0
            for frame, weight in enumerate(mass):
                if before + weight >= threshold:
                    expected.append((frame + (threshold - before) / weight) * 0.02)
                    break
                before += weight
        out = refine_from_ottc_mass(
            torch.tensor(mass), torch.linspace(0, duration, n + 1)[1:-1], duration
        )
        torch.testing.assert_close(out.global_times, torch.tensor(expected))

    def test_mixed_acceptance_never_crosses(self):
        torch.manual_seed(3)
        for _ in range(30):
            coarse = torch.rand(20).sort().values
            out = refine_from_ottc_mass(torch.rand(50), coarse, 1.0)
            self.assertTrue((out.times.diff() >= 0).all())
            self.assertTrue((out.offsets.abs() <= 0.020001).all())
            torch.testing.assert_close(out.times[~out.accepted], coarse[~out.accepted])

    def test_zero_mass_cells_one_phone_beta_and_validation(self):
        out = refine_from_ottc_mass(torch.tensor([0.0, 1.0, 0.0]), torch.empty(0), 0.06)
        self.assertEqual(out.times.numel(), 0)
        out = refine_from_ottc_mass(
            torch.ones(2),
            torch.tensor([0.05]),
            0.2,
            target_mass=torch.tensor([1.0, 3.0]),
        )
        torch.testing.assert_close(out.times, torch.tensor([0.05]))
        with self.assertRaises(ValueError):
            refine_from_ottc_mass(torch.zeros(2), torch.tensor([0.1]), 0.2)
        with self.assertRaises(ValueError):
            refine_from_ottc_mass(
                torch.ones(2),
                torch.tensor([0.1]),
                0.2,
                target_mass=torch.tensor([0.0, 1.0]),
            )


class CumulativeMassTest(unittest.TestCase):
    def test_original_cif_example_and_multiple_crossings(self):
        mass = torch.tensor([0.2, 0.9, 0.6, 0.6], dtype=torch.double)
        out = mass_quantiles(
            mass,
            torch.arange(5, dtype=torch.double),
            torch.tensor([1 / 2.3, 2 / 2.3], dtype=torch.double),
        )
        torch.testing.assert_close(
            out, torch.tensor([1 + 0.8 / 0.9, 3 + 0.3 / 0.6], dtype=torch.double)
        )
        # Three thresholds can fall in a single cell, unlike single-fire loops.
        out = mass_quantiles(
            torch.tensor([0.0, 4.0, 0.0]),
            torch.arange(4.0),
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        torch.testing.assert_close(out, torch.tensor([0.0, 1.25, 1.5, 1.75, 3.0]))

    def test_gradients_and_scale_invariance(self):
        mass = torch.tensor([[0.3, 0.7, 0.8]], dtype=torch.double, requires_grad=True)
        edges = torch.tensor([[0.0, 0.02, 0.04, 0.06]], dtype=torch.double)
        q = torch.tensor([[0.3, 0.65]], dtype=torch.double)
        self.assertTrue(
            torch.autograd.gradcheck(lambda x: mass_quantiles(x, edges, q), (mass,))
        )
        torch.testing.assert_close(
            mass_quantiles(mass, edges, q), mass_quantiles(mass * 7, edges, q)
        )

    def test_reject_invalid(self):
        for mass in ([0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                mass_quantiles(
                    torch.tensor(mass), torch.arange(3.0), torch.tensor([0.5])
                )
        with self.assertRaises(ValueError):
            mass_quantiles(torch.ones(2), torch.arange(3.0), torch.tensor([1.1]))


class RefinerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = CIFBoundaryRefiner(6, 5, hidden_dim=16, phone_dim=4)
        self.features = torch.randn(10, 6)
        self.coarse = torch.tensor([0.04, 0.041, 0.08, 0.14])
        self.phones = torch.tensor([1, 1, 3, 2, 1])  # Repeated tokens retained.

    def run_model(self):
        return self.model(self.features, self.coarse, self.phones, 0.2)

    def test_identity_and_shared_edges(self):
        out = self.run_model()
        torch.testing.assert_close(out.times, self.coarse, rtol=0, atol=2e-8)
        self.assertTrue(
            (out.window_edges[:-1, -1] <= out.window_edges[1:, 0] + 1e-8).all()
        )
        self.assertEqual(len(out.times), 4)

    def test_extreme_density_stays_ordered_and_bounded(self):
        with torch.no_grad():
            self.model.density[-1].weight.normal_(0, 100)
        out = self.run_model()
        self.assertTrue(torch.isfinite(out.times).all())
        self.assertTrue((out.times.diff() >= 0).all())
        self.assertTrue((out.offsets.abs() <= 0.020001).all())
        self.assertTrue((out.times >= out.window_edges[:, 0]).all())
        self.assertTrue((out.times <= out.window_edges[:, -1]).all())

    def test_empty_degenerate_and_single_frame(self):
        out = self.model(self.features[:1], torch.empty(0), torch.tensor([1]), 0.2)
        self.assertEqual(out.mass.shape, (0, 8))
        self.assertEqual(boundary_loss(out, torch.empty(0)).item(), 0.0)
        times = torch.tensor([0.0, 0.1, 0.1, 0.2])
        out = self.model(self.features[:1], times, self.phones, 0.2)
        torch.testing.assert_close(out.times, times)
        with self.assertRaises(ValueError):
            self.model(self.features, self.coarse.flip(0), self.phones, 0.2)

    def test_explicit_clock(self):
        out = self.model(
            self.features,
            self.coarse,
            self.phones,
            0.2,
            frame_edges=torch.linspace(-0.002, 0.198, 11),
        )
        torch.testing.assert_close(out.times, self.coarse)
        with self.assertRaises(ValueError):
            self.model(
                self.features,
                self.coarse,
                self.phones,
                0.2,
                frame_edges=torch.zeros(11),
            )

    def test_learning_fractional_time_and_checkpoint(self):
        # Deliberately non-grid reference; train ONLY the postprocessor.
        coarse, phones = torch.tensor([0.1]), torch.tensor([1, 2])
        gold = torch.tensor([0.1067])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        def predict():
            return self.model(self.features, coarse, phones, 0.2)

        initial = boundary_loss(predict(), gold).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = boundary_loss(predict(), gold)
            loss.backward()
            optimizer.step()
        out = predict()
        self.assertLess(boundary_loss(out, gold).item(), initial / 20)
        self.assertLess(abs(out.times.item() - 0.1067), 0.0005)
        buf = io.BytesIO()
        torch.save(dict(config=self.model.config, state=self.model.state_dict()), buf)
        buf.seek(0)
        saved = torch.load(buf, weights_only=True)
        other = CIFBoundaryRefiner(**saved["config"])
        other.load_state_dict(saved["state"])
        torch.testing.assert_close(
            out.times, other(self.features, coarse, phones, 0.2).times
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
