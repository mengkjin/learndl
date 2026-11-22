"""Causal factor encoders, conditional residuals and compact result delivery."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from src.res.algo.nn.model.factor_sequence.contextual_v4 import FactorContextualV4
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual

__all__ = ["ContextualSequenceTests"]


def example(factors=3):
    stock = torch.randn(7, 60, 4 * factors)
    stock[..., factors : 2 * factors] = 1
    perf = torch.randn(1, 60, 4 * factors)
    perf[..., 1::2] = 1
    return stock, perf


class ContextualSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_parameter_budget_and_train_mode_identity(self):
        stock, perf = example(33)
        temporal = FactorResidual(132, mode="temporal")
        torch.nn.init.normal_(temporal.output.weight)
        for variant, count in (("shared", 23979), ("private", 28003), ("graph", 24975)):
            model = FactorContextualV4([132, 132], variant=variant).train()
            model.load_state_dict(temporal.state_dict(), strict=False)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            rng = torch.get_rng_state()
            expected = temporal(stock)[0]
            torch.set_rng_state(rng)
            score, other = model([stock.unsqueeze(2), perf.unsqueeze(2)])
            torch.testing.assert_close(score, expected, rtol=0, atol=0)
            if variant == "shared":
                self.assertEqual(other["loss_context_shrinkage"].item(), 0)
            torch.testing.assert_close(
                other["linear_score"] + other["residual_score"], score
            )
            missing = perf.clone()
            missing[..., 1::2] = 0
            torch.nn.init.normal_(model.factor_film.weight)
            score, other = model([stock, missing])
            torch.testing.assert_close(score, other["temporal_score"], rtol=0, atol=0)
            self.assertEqual(other["loss_context_shrinkage"].item(), 0)

    def test_stock_and_performance_order_maturity_and_causality(self):
        stock, perf = example()
        model = FactorContextualV4([12, 12], dropout=0, context_dropout=0).eval()
        torch.nn.init.normal_(model.factor_film.weight)
        state, ready, _ = model._encode_performance(perf)
        for f in range(3):
            expected = model.shared_gru(perf[:, :, 4 * f : 4 * (f + 1)])[1][0, 0]
            torch.testing.assert_close(state[f], expected)
        self.assertTrue(ready.all())
        changed = perf.clone()
        changed[:, 0, 0] += 5
        other_state = model._encode_performance(changed)[0]
        torch.testing.assert_close(other_state[1:], state[1:])
        # Same values, reverse order. A sequence encoder can distinguish the histories.
        reversed_state = model._encode_performance(perf.flip(1))[0]
        self.assertFalse(torch.allclose(state, reversed_state))
        perf[:, :41, 3] = 0
        perf[:, :41, 2] = float("nan")
        score, other = model([stock, perf])
        self.assertEqual(other["performance_ready"].tolist(), [False, True, True])
        torch.testing.assert_close(other["film_gamma"][0], torch.zeros(8))
        self.assertTrue(torch.isfinite(score).all())
        full = torch.cat([perf, torch.randn(1, 4, 12)], dim=1)
        torch.testing.assert_close(score, model([stock, full[:, :60]])[0])
        # Only factor zero is observed now and its context is immature.
        stock[:, -1, 4:6] = 0
        _, other = model([stock, perf])
        torch.testing.assert_close(
            other["context_correction"], torch.zeros(len(stock), 1)
        )
        with self.assertRaises(ValueError):
            model([stock, perf.expand(len(stock), -1, -1)])
        with self.assertRaises(ValueError):
            model([stock, full])

    def test_graph_neighbors_mask_and_permutation(self):
        model = FactorContextualV4([12, 12], variant="graph")
        state = torch.randn(3, 16)
        ready = torch.tensor([True, False, True])
        output, _, _, attention = model.factor_graph(state, ready)
        torch.testing.assert_close(
            attention, torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        for sparse in (
            torch.zeros(3, dtype=torch.bool),
            torch.tensor([True, False, False]),
        ):
            _, _, delta, weights = model.factor_graph(state, sparse)
            torch.testing.assert_close(delta, torch.zeros_like(delta))
            torch.testing.assert_close(weights, torch.zeros_like(weights))
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            model.factor_graph.identity.copy_(model.factor_graph.identity[permutation])
        swapped, _, _, weights = model.factor_graph(
            state[permutation], ready[permutation]
        )
        torch.testing.assert_close(swapped, output[permutation])
        torch.testing.assert_close(weights, attention[permutation][:, permutation])

    def test_loss_gradients_and_saved_weights(self):
        from src.res.model.util.core import BatchData, BatchInput, BatchOutput
        from src.res.model.util.metric.functions import LossFunction
        from src.res.model.util.metric.aggregator import LossAggregator

        stock, perf = example()
        args = [stock, perf]
        target = torch.randn(len(stock), 1)
        batch = BatchInput(
            args,
            target,
            None,
            torch.zeros(len(stock), 2, dtype=torch.long),
            torch.ones(len(stock), dtype=torch.bool),
            np.array([20170103]),
            np.arange(len(stock)),
        )
        for variant in ("shared", "private", "graph"):
            model = FactorContextualV4(
                [12, 12], variant=variant, context_dropout=0, dropout=0
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for step in range(3):
                optimizer.zero_grad()
                result = model(args)
                losses = LossFunction({"mse": {}}).losses(
                    BatchData(batch, BatchOutput(result))
                )
                self.assertEqual(set(losses), {"mse", "loss_context_shrinkage"})
                LossAggregator().loss(losses).backward()
                if step == 0:
                    grad = model.factor_film.weight.grad
                    assert grad is not None
                    self.assertGreater(grad.abs().sum().item(), 0)
                optimizer.step()
            named = dict(model.named_parameters())
            checks = [
                "shared_gru.weight_ih_l0",
                "value_encoder.weight",
                "readout.weight",
            ]
            if variant == "private":
                self.assertEqual(
                    len({id(g.weight_ih_l0) for g in model.private_grus}), 3
                )
                checks += [f"private_grus.{f}.weight_ih_l0" for f in range(3)]
            elif variant == "graph":
                checks += [
                    "factor_graph.identity",
                    "factor_graph.query.weight",
                    "factor_graph.output.weight",
                ]
            for key in checks:
                grad = named[key].grad
                assert grad is not None
                self.assertTrue(grad.isfinite().all(), key)
                self.assertGreater(grad.abs().sum().item(), 0, key)
            saved = BytesIO()
            torch.save(model.state_dict(), saved)
            saved.seek(0)
            clone = FactorContextualV4([12, 12], variant=variant).eval()
            clone.load_state_dict(torch.load(saved, weights_only=True))
            score, other = model.eval()(args)
            torch.testing.assert_close(clone(args)[0], score)
            self.assertLessEqual(other["film_gamma"].abs().max().item(), 0.200001)
            self.assertLessEqual(other["film_beta"].abs().max().item(), 0.100001)

    def test_context_summary_uses_model_result_collector(self):
        from src.automation.artifact_exchange.research import model_result_specs
        from src.res.model.callback.test.factor_residual import (
            FactorResidualDiagnostics,
        )

        model = FactorContextualV4([12, 12], variant="graph")
        _, other = model(example())
        row = FactorResidualDiagnostics.context_summary(20170103, other)
        self.assertEqual(row["ready_factors"], 3)
        self.assertEqual(row["attention_rows"], 3)
        entropy = row["attention_entropy"]
        assert entropy is not None
        self.assertGreater(entropy, 0)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "results/factor_context_v4/daily_context.csv"
            report.parent.mkdir(parents=True)
            report.write_text("date,ready_factors\n20170103,3\n")
            specs = model_result_specs(root, "factorseq_v4_graph")
            self.assertTrue(
                any(
                    spec.members is not None
                    and "factor_context_v4/daily_context.csv" in spec.members
                    for spec in specs
                )
            )

    def test_comparison_pairs_real_dates_stocks_and_labels(self):
        from src.res.model.analytic.factor_sequence.compare_v4 import compare_v4

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            roots = {}
            for variant in ("temporal", "shared", "private", "graph"):
                root = folder / variant
                roots[variant] = root
                version = "v3" if variant == "temporal" else "v4"
                config = {
                    "model.name": f"factorseq_{version}_{variant}",
                    "input.factor_sequence.dataset": "/frozen/dataset",
                    "model.labels": ["std_lag1_10"],
                    "train.factor_sequence.folds": [{"test": [20170103]}],
                }
                (root / "configs").mkdir(parents=True)
                (root / "configs/model.yaml").write_text(yaml.safe_dump(config))
                predictions = root / "snapshot/factor_residual"
                predictions.mkdir(parents=True)
                pd.DataFrame(
                    dict(
                        secid=[1, 2, 3],
                        label=[1.0, 2.0, 3.0],
                        pred=[3.0, 2.0, 1.0]
                        if variant == "temporal"
                        else [1.0, 2.0, 3.0],
                        linear=[1.0, 2.0, 3.0],
                        residual=[0.1, 0.3, 0.2],
                        coverage=[1.0, 1.0, 1.0],
                    )
                ).to_feather(predictions / "20170103.feather")
                metrics = root / "snapshot/metrics"
                metrics.mkdir()
                with pd.ExcelWriter(
                    metrics / "model.0.20161230.xlsx", engine="openpyxl"
                ) as writer:
                    for split in ("train", "valid"):
                        pd.DataFrame(dict(epoch=[0, 1], rankic=[0.05, 0.1])).to_excel(
                            writer, sheet_name=split + "_epoch_totals", index=False
                        )
            output = folder / "comparison"
            compare_v4(**roots, output=output)
            daily = pd.read_csv(output / "shared_daily.csv")
            self.assertAlmostEqual(
                daily.loc[daily.group == "all", "delta_ic"].item(), 2.0
            )
            self.assertTrue((output / "graph_annual.csv").is_file())
            self.assertTrue((output / "temporal/epoch_summary.csv").is_file())
            bad = roots["graph"] / "snapshot/factor_residual/20170103.feather"
            pd.read_feather(bad).assign(label=[1.0, 2.0, 4.0]).to_feather(bad)
            with self.assertRaisesRegex(ValueError, "Stock/label mismatch"):
                compare_v4(**roots, output=folder / "mismatched")


if __name__ == "__main__":
    unittest.main()
