"""Tests for v5 market contributions, the frozen schedule, and paired reporting."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.res.algo.nn.model.factor_sequence.contextual_v4 import FactorContextualV4
from src.res.algo.nn.model.factor_sequence.contextual_v5 import FactorContextualV5
from src.res.model.analytic.factor_sequence.compare_v5 import (
    _event_table,
    _hac_interval,
    _snapshot_rows,
)
from src.res.model.experiment.factor_sequence.repaired_schedule import (
    GRAPH_LR_PROBE_NODES,
)
from src.res.model.experiment.factor_sequence.worklist import residual_config
from src.res.model.util.lifecycle import ModelLifecycle

__all__ = ["V5ModelTests", "V5ScheduleTests", "V5ReportTests"]


def inputs(k=33):
    stock = torch.randn(5, 60, 4 * k)
    stock[..., k : 2 * k] = 1
    perf = torch.randn(1, 60, k, 2, 2)
    perf[..., 1] = 1
    market = torch.zeros(1, 1, 30)
    market[..., 1::2] = 1
    market[..., [2, 10]] = 1  # first and third events have capped ages
    market[..., 4] = 1  # second event triggers today
    return [stock, perf.flatten(2), market]


class V5ModelTests(unittest.TestCase):
    def test_parameters_and_exact_parent_prediction(self):
        v4 = FactorContextualV4([132, 132], variant="graph").eval()
        torch.nn.init.normal_(v4.factor_film.weight, std=0.1)
        x = inputs()
        with torch.no_grad():
            base, diagnostics = v4(x[:2])
            for variant, n in (("market", 25141), ("market_events", 25186)):
                model = FactorContextualV5([132, 132, 30], variant=variant).eval()
                state = model.load_state_dict(v4.state_dict(), strict=False)
                self.assertFalse(state.unexpected_keys)
                self.assertEqual(
                    len(state.missing_keys), 3 if variant == "market" else 5
                )
                self.assertEqual(sum(p.numel() for p in model.parameters()), n)
                score, other = model([value.unsqueeze(2) for value in x])
                torch.testing.assert_close(score, base, rtol=0, atol=0)
                torch.testing.assert_close(
                    other["loss_context_shrinkage"],
                    diagnostics["loss_context_shrinkage"],
                )

    def test_branch_isolation_missingness_loss_and_round_trip(self):
        torch.manual_seed(17)
        x = inputs(k=3)
        model = FactorContextualV5(
            [12, 12, 30], variant="market_events", dropout=0, context_dropout=0
        )
        x[1][..., 1::2] = 0  # market still adjusts valid raw factor values
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            score, other = model(x)
            (score.square().mean() + other["loss_context_shrinkage"]).backward()
            optimizer.step()
        assert model.event_gate is not None
        for layer in (
            model.market_gate.project,
            model.market_gate.output,
            model.stock_market_gate,
            model.event_gate.project,
            model.event_gate.output,
        ):
            assert layer.weight.grad is not None
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        model.eval()
        x[1][..., 1::2] = 1
        score, other = model(x)
        torch.testing.assert_close(
            score,
            other["graph_score"]
            + other["market_factor_correction"]
            + other["market_stock_correction"]
            + other["event_factor_correction"],
        )
        self.assertTrue(other["market_factor_multiplier"].ge(0.8).all())
        self.assertTrue(other["market_factor_multiplier"].le(1.2).all())
        missing_event = x[2].clone()
        missing_event[..., :12] = 0
        _, changed = model([*x[:2], missing_event])
        for key in (
            "graph_score",
            "factor_attention",
            "market_factor_correction",
            "market_stock_correction",
        ):
            torch.testing.assert_close(other[key], changed[key], atol=0, rtol=0)
        self.assertTrue(changed["event_factor_correction"].eq(0).all())
        score_missing, _ = model([*x[:2], torch.zeros_like(x[2])])
        torch.testing.assert_close(score_missing, other["graph_score"], atol=0, rtol=0)
        clone = FactorContextualV5([12, 12, 30], variant="market_events").eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(clone(x)[0], score)
        market_only = FactorContextualV5([12, 12, 30], variant="market").eval()
        market_only.load_state_dict(
            {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith("event_gate.")
            }
        )
        torch.testing.assert_close(
            market_only(x)[0], market_only([*x[:2], missing_event])[0], atol=0, rtol=0
        )
        x[0][:, -1, 3:6] = 0
        _, no_values = model(x)
        self.assertTrue(no_values["market_factor_correction"].eq(0).all())
        self.assertTrue(no_values["event_factor_correction"].eq(0).all())


class V5ScheduleTests(unittest.TestCase):
    def test_config_and_probe_scope(self):
        dates = list(range(20000000, 20000037))
        dates[:19] = list(range(20161230, 20161249))
        dates[18:36] = list(range(20210827, 20210845))
        dates[36] = GRAPH_LR_PROBE_NODES[-1]
        folds = [
            dict(model_date=d, test=[d + 1], train=[d - 9], valid=[d - 4])
            for d in dates
        ]
        v4 = residual_config(
            "/dataset", "graph", "factorseq_contextual_v4", folds, "graph"
        )
        v5 = residual_config(
            "/dataset", "market", "factorseq_contextual_v5", folds, "market"
        )
        self.assertEqual(
            v5["input.sequence.lens"],
            {"factorseq": 60, "factorperf": 60, "factormarket": 1},
        )
        self.assertEqual(
            v4["train.factor_sequence.folds"], v5["train.factor_sequence.folds"]
        )
        self.assertEqual(v4["train.trainer.learn_rate"], v5["train.trainer.learn_rate"])
        probe = residual_config(
            "/dataset", "probe", "factorseq_contextual_v4", folds, "graph_lr_probe"
        )
        self.assertEqual(probe["train.trainer.learn_rate"]["base"], 0.0003)
        self.assertEqual(
            probe["train.factor_sequence.fit_nodes"], list(GRAPH_LR_PROBE_NODES)
        )
        from src.res.model.util.config.config import ModelConfig

        def resolve_fixture(source, model, name, mode):
            self.assertEqual(source, "repaired_20261117")
            return residual_config("/dataset", name, model, folds, mode)

        with patch(
            "src.res.model.experiment.factor_sequence.worklist.resolve_schedule",
            side_effect=resolve_fixture,
        ):
            config_v5 = ModelConfig(
                schedule_name="factorseq_v5_graph_market_events",
                stage=0,
                override={"env.short_test": False},
            )
            config_probe = ModelConfig(
                schedule_name="factorseq_v4_graph_lr_probe",
                stage=1,
                override={"env.short_test": False},
            )
        self.assertEqual(config_v5.seq_lens, v5["input.sequence.lens"])
        self.assertEqual(config_v5.trainer_learn_rate["base"], 0.001)
        self.assertEqual(config_probe.trainer_learn_rate["base"], 0.0003)
        config_probe.parser_stage(1)
        self.assertEqual(config_probe.queue_of_stages, ["data", "fit"])
        self.assertEqual(
            config_probe.Param.get("train.factor_sequence.fit_nodes"),
            list(GRAPH_LR_PROBE_NODES),
        )
        config = SimpleNamespace(Param=probe, queue_of_stages=["data", "fit"])
        loaded = np.array([dates[0], *[d + 1 for d in dates]])
        with (
            patch(
                "src.res.model.util.data.factor_sequence_plan.validate_folds",
                return_value=(dates, loaded[1:]),
            ),
            patch.object(ModelLifecycle, "selected_nodes", return_value=dates[::18]),
        ):
            self.assertEqual(
                ModelLifecycle.fold_nodes(config, folds, loaded)[0].tolist(),
                list(GRAPH_LR_PROBE_NODES),
            )
            config.queue_of_stages.append("test")
            with self.assertRaisesRegex(ValueError, "fit-only"):
                ModelLifecycle.fold_nodes(config, folds, loaded)


class V5ReportTests(unittest.TestCase):
    def test_event_segments_and_constant_hac(self):
        from src.data.preprocess.factor_sequence.schema import EVENTS

        daily = pd.DataFrame({"date": range(8), "delta_ic": [0.1] * 8})
        context = pd.DataFrame({"date": daily.date})
        for name in EVENTS:
            context[name] = [0, 1, 1, 0, np.nan, 0, 1, 0]
            context[name + "__age"] = np.log1p(
                [np.nan, 0, 0, 1, np.nan, 0, 0, 1]
            ) / np.log1p(252)
        table = _event_table(context, daily)
        self.assertEqual(table.iloc[0]["区间触发段数"], 2)
        self.assertEqual(table.iloc[0]["天数"], 5)
        np.testing.assert_allclose(_hac_interval([0.1] * 21), [0.1, 0.1])

    def test_score_attribution_and_label_check(self):
        with TemporaryDirectory() as temp:
            a, b = (Path(temp) / k for k in ("a", "b"))
            for root in (a, b):
                (root / "snapshot/factor_residual").mkdir(parents=True)
            left = pd.DataFrame(
                {"secid": range(51), "label": np.arange(51, dtype=float)}
            )
            left.to_feather(a / "snapshot/factor_residual/20170101.feather")
            right = left.assign(
                graph_score=left.label,
                market_factor_correction=-2 * left.label,
                market_stock_correction=0.0,
                event_factor_correction=0.0,
                pred=-left.label,
            )
            file = b / "snapshot/factor_residual/20170101.feather"
            right.to_feather(file)
            daily = pd.DataFrame({"delta_ic": [-2.0]}, index=[20170101])
            (row,) = _snapshot_rows(a, b, daily)
            self.assertAlmostEqual(row["same_weights_delta_std_ic"], -2)
            self.assertEqual(row["top50_replaced"], 1)
            right.loc[0, "label"] = -100
            right.to_feather(file)
            with self.assertRaisesRegex(ValueError, "labels differ"):
                list(_snapshot_rows(a, b, daily))
