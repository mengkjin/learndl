"""Fixed summaries, bounded residual gates and ordinary training loss integration."""

from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
import torch

from src.res.algo.nn.model.factor_sequence.performance_gate import PerformanceGate
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual

__all__ = ["PerformanceGateTests", "ContextualV4Tests"]


def inputs(k=3, days=120):
    stock = torch.randn(7, 60, 4 * k)
    stock[..., k : 2 * k] = 1
    perf = torch.randn(1, days, k, 2, 2)
    perf[..., 1] = 1
    return stock, perf.flatten(2)


class PerformanceGateTests(unittest.TestCase):
    def test_summaries_against_independent_numpy(self):
        _, perf = inputs()
        records = perf.reshape(120, 3, 2, 2)
        records[::3, 0, 0, 1] = 0
        records[::3, 0, 0, 0] = float("nan")
        records[:110, 1, 1, 1] = 0
        gate = PerformanceGate(3)
        features, ready = gate.summarize(perf)
        self.assertEqual(ready.tolist(), [True, False, True])
        expected = []
        for factor in range(3):
            row = []
            for metric in range(2):
                value = records[:, factor, metric, 0].numpy()
                valid = (records[:, factor, metric, 1].numpy() == 1) & np.isfinite(
                    value
                )
                means, coverages = [], []
                for half_life in (5, 20, 60):
                    w = 2.0 ** (-np.arange(119, -1, -1) / half_life)
                    means.append(np.sum(w[valid] * value[valid]) / np.sum(w[valid]))
                    coverages.append(np.sum(w[valid]) / np.sum(w))
                row.extend(means + coverages + [means[0] - means[-1]])
            expected.append(row)
        np.testing.assert_allclose(features.numpy(), expected, atol=1e-6)
        # A placeholder's magnitude has no effect when its validity is false.
        records[::3, 0, 0, 0] = 999999
        torch.testing.assert_close(features, gate.summarize(perf)[0])

    def test_missing_maturity_bounds_and_factor_permutation(self):
        _, perf = inputs()
        gate = PerformanceGate(3)
        with torch.no_grad():
            gate.output.bias.fill_(100)
        records = perf.reshape(120, 3, 2, 2)
        records[:, 0, :, 1] = 0
        records[:, 0, :, 0] = float("nan")
        records[:101, 1, :, 1] = 0  # only 19 observations
        gates, features, ready = gate(perf)
        torch.testing.assert_close(gates, torch.tensor([1.0, 1.0, 1.2]))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(ready.tolist(), [False, False, True])
        permutation = [2, 0, 1]
        shuffled = records[:, permutation].reshape_as(perf)
        torch.testing.assert_close(gate(shuffled)[0], gates[permutation])
        with torch.no_grad():
            gate.output.bias.fill_(-100)
        torch.testing.assert_close(gate(perf)[0], torch.tensor([1.0, 1.0, 0.8]))

    def test_prefix_causality_and_window_boundary(self):
        _, perf = inputs(days=140)
        gate = PerformanceGate(3)
        before = gate.summarize(perf[:, :120])[0]
        perf[:, 120:, ::2] = 1e9
        torch.testing.assert_close(before, gate.summarize(perf[:, :120])[0])
        # Scale-aware summaries over the supplied trailing window, not hidden state.
        parts = torch.cat([perf[:, :37], perf[:, 37:90], perf[:, 90:]], dim=1)
        torch.testing.assert_close(
            gate.summarize(parts[:, -120:])[0], gate.summarize(perf[:, -120:])[0]
        )
        for half_lives in ((0, 20), (20, 5), (5, float("nan"))):
            with self.assertRaises(ValueError):
                PerformanceGate(3, half_lives=half_lives)


class ContextualV4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_temporal_identity_and_bounded_contribution(self):
        temporal = FactorResidual(12, mode="temporal", dropout=0)
        model = FactorResidual([12, 12], mode="contextual_v4", dropout=0)
        torch.nn.init.normal_(temporal.output.weight)
        incompatible = model.load_state_dict(temporal.state_dict(), strict=False)
        self.assertTrue(
            all(k.startswith("performance_gate.") for k in incompatible.missing_keys)
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        stock, perf = inputs()
        base = temporal(stock)[0]
        score, other = model([stock, perf])
        torch.testing.assert_close(score, base)
        self.assertEqual(other["loss_context_shrinkage"].item(), 0)
        with torch.no_grad():
            model.performance_gate.output.bias.fill_(100)
        score, other = model([stock, perf])
        expected = torch.nn.functional.linear(
            stock[:, -1, :3] * 0.2, model.linear.weight
        )
        torch.testing.assert_close(score, base + expected)
        torch.testing.assert_close(
            other["linear_score"] + other["residual_score"], score
        )
        self.assertAlmostEqual(other["loss_context_shrinkage"].item(), 0.0004, places=7)
        # Last-day invalid values cannot acquire a context correction.
        stock[:, -1, 3:6] = 0
        torch.testing.assert_close(
            model([stock, perf])[1]["context_correction"], torch.zeros(7, 1)
        )

    def test_loss_path_training_and_serialization(self):
        from src.res.model.util.metric.functions import LossFunction
        from src.res.model.util.metric.aggregator import LossAggregator
        from src.res.model.util.core import BatchData, BatchInput, BatchOutput

        model = FactorResidual([12, 12], mode="contextual_v4", dropout=0)
        stock, perf = inputs()
        target = torch.randn(7, 1)
        batch_input = BatchInput(
            [stock, perf],
            target,
            None,
            torch.zeros(7, 2, dtype=torch.long),
            torch.ones(7, dtype=torch.bool),
            np.array([20170103]),
            np.arange(7),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(4):
            optimizer.zero_grad()
            score, other = model([stock, perf])
            data = BatchData(batch_input, BatchOutput((score, other)))
            loss_function = LossFunction({"mse": {}})
            losses = loss_function.losses(data)
            self.assertEqual(set(losses), {"mse", "loss_context_shrinkage"})
            LossAggregator().loss(losses).backward()
            optimizer.step()
        for layer in (
            model.performance_gate.hidden,
            model.performance_gate.output,
            model.output,
            model.project,
        ):
            assert layer.weight.grad is not None
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
        buffer = BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = FactorResidual([12, 12], mode="contextual_v4", dropout=0)
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model([stock, perf])[0], clone([stock, perf])[0])
        perf[..., 1::2] = 0
        score, other = clone([stock, perf])
        torch.testing.assert_close(score, other["temporal_score"])
        self.assertTrue(torch.isfinite(score).all())

    def test_capacity_and_contract(self):
        model = FactorResidual([132, 132], mode="contextual_v4")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 22404)
        self.assertEqual(
            sum(p.numel() for p in model.performance_gate.parameters()), 129
        )
        self.assertFalse(hasattr(model, "market"))
        self.assertFalse(hasattr(model, "film"))
        stock, perf = inputs(k=33)
        self.assertEqual(
            model([stock.unsqueeze(2), perf.unsqueeze(2)])[0].shape, (7, 1)
        )
        for streams in (
            [stock],
            [stock, perf.expand(7, -1, -1)],
            [stock, perf[:, :59]],
            [stock, perf, perf],
        ):
            with self.assertRaises(ValueError):
                model(streams)
        with self.assertRaises(ValueError):
            FactorResidual([132, 396], mode="contextual_v4")


if __name__ == "__main__":
    unittest.main()
