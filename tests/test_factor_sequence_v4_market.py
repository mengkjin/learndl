"""Market/event separation, availability, nested initialization and training checks."""

from __future__ import annotations

from io import BytesIO
import math
import unittest

import numpy as np
import pandas as pd
import torch

from src.data.preprocess.factor_sequence.context_audit import (
    MARKET_NAMES,
    RISK_NAMES,
    candidate_market,
)
from src.data.preprocess.factor_sequence.numeric import pack
from src.data.preprocess.factor_sequence.schema import EVENTS
from src.res.algo.nn.model.factor_sequence.market_gate import EventGate, MarketGate
from src.res.algo.nn.model.factor_sequence.residual import FactorResidual

__all__ = ["MarketEncodingTests", "V4MarketModelTests"]


def sample(k=3):
    stock = torch.randn(7, 60, 4 * k)
    stock[..., k : 2 * k] = 1
    perf = torch.randn(1, 120, 4 * k)
    perf[..., 1::2] = 1
    market = torch.zeros(1, 1, 30)
    market[..., 12::2] = torch.arange(9) / 10
    market[..., 12] = 1  # market_up
    market[..., 13::2] = 1
    # An observed event ten trading days ago; other events still unknown.
    market[..., 1] = market[..., 3] = 1
    market[..., 2] = math.log1p(10) / math.log1p(252)
    return stock, perf, market


class MarketEncodingTests(unittest.TestCase):
    def test_continuous_feature_order_and_invalid_values(self):
        _, _, market = sample()
        gate = MarketGate(3)
        features, ready = gate.encode(market)
        self.assertTrue(ready)
        self.assertEqual(tuple(MARKET_NAMES[6:]), ("market_up", *RISK_NAMES))
        torch.testing.assert_close(features, market.flatten()[12:])
        dirty = market.clone()
        dirty[..., :12] = float("nan")
        torch.testing.assert_close(gate.encode(dirty)[0], features)
        dirty[..., 14] = float("inf")  # finite source validity is not sufficient
        dirty[..., 17] = 0
        dirty[..., 16] = float("nan")
        masked = gate.encode(dirty)[0]
        torch.testing.assert_close(masked[2:6], torch.zeros(4))
        dirty[..., 13::2] = 0
        self.assertFalse(gate.encode(dirty)[1])
        with torch.no_grad():
            gate.output.weight.fill_(1)
        torch.testing.assert_close(gate(dirty)[0], torch.zeros(3))

    def test_event_age_unknown_trigger_decay_cap_and_source_gap(self):
        _, _, market = sample()
        gate = EventGate(3)
        features, ready = gate.encode(market)
        self.assertTrue(ready)
        torch.testing.assert_close(
            features[:4], torch.tensor([0, 1, math.exp(-0.5), 1])
        )
        # Continuous market data cannot enter event features or response.
        altered = market.clone()
        altered[..., 12:] = float("nan")
        torch.testing.assert_close(gate.encode(altered)[0], features)
        # Unknown age is never interpreted as age zero, even with zero placeholder.
        altered[..., :12] = 0
        altered[..., 1] = 1  # source confirms no trigger
        self.assertFalse(gate.encode(altered)[1])
        torch.testing.assert_close(
            gate.encode(altered)[0][:4], torch.tensor([0.0, 1.0, 0.0, 0.0])
        )
        altered[..., 0] = 1  # a known trigger can stand without known age
        self.assertTrue(gate.encode(altered)[1])
        # Same-day multiple events occupy distinct fields; no last-event overwrite.
        altered[..., :12] = torch.tensor([1.0, 1.0, 0.0, 1.0]).repeat(3)
        torch.testing.assert_close(gate.encode(altered)[0], torch.ones(12))
        # At the storage cap (252 or older), stop rather than retain a fixed tail.
        altered[..., :12] = torch.tensor([0.0, 1.0, 1.0, 1.0]).repeat(3)
        self.assertFalse(gate.encode(altered)[1])
        # An invalid event source also invalidates a seemingly present age.
        altered[..., :12] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(3)
        self.assertFalse(gate.encode(altered)[1])

    def test_event_response_decays_even_if_network_reads_only_validity(self):
        gate = EventGate(3)
        with torch.no_grad():
            gate.project.weight.zero_()
            gate.project.weight[0, 1] = 1
            gate.output.weight.fill_(1)
        market = torch.zeros(1, 1, 30)
        market[..., 1] = market[..., 3] = 1
        start = gate(market)[0]
        market[..., 2] = math.log1p(20) / math.log1p(252)
        torch.testing.assert_close(gate(market)[0], start * math.exp(-1))

    def test_real_market_contract_preserves_prefix_and_event_states(self):
        raw = pd.DataFrame(
            {name: np.arange(70, dtype=float) for name in RISK_NAMES},
            index=pd.bdate_range("2017-01-02", periods=70),
        )
        for event in EVENTS:
            raw[event] = 0.0
        raw["market_return"] = 0.01
        raw.loc[raw.index[4], EVENTS[0]] = 1
        raw.loc[raw.index[9], EVENTS[0]] = np.nan
        raw.loc[raw.index[12], EVENTS[0]] = 1
        prefix = candidate_market(raw.iloc[:20])
        full = candidate_market(raw)
        pd.testing.assert_frame_equal(prefix, full.iloc[:20])
        packed, names = pack({name: full[name].to_numpy() for name in MARKET_NAMES})
        self.assertEqual(
            names, [f"{n}{s}" for n in MARKET_NAMES for s in ("", "__valid")]
        )
        gate = EventGate(3)
        for date, expected in (
            (3, False),
            (4, True),
            (8, True),
            (9, False),
            (11, False),
            (12, True),
        ):
            market = torch.as_tensor(packed[date : date + 1], dtype=torch.float32)[None]
            self.assertEqual(bool(gate.encode(market)[1]), expected)


class V4MarketModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_nested_identity_and_parameter_counts(self):
        inputs = sample(k=33)
        base = FactorResidual([132, 132], mode="contextual_v4", dropout=0)
        torch.nn.init.normal_(base.output.weight)
        torch.nn.init.normal_(base.performance_gate.output.weight)
        for mode, expected_count, prefix in (
            ("contextual_v4_market", 22506, "market_gate."),
            ("contextual_v4_events", 22551, "event_gate."),
        ):
            model = FactorResidual([132, 132, 30], mode=mode, dropout=0)
            mismatch = model.load_state_dict(base.state_dict(), strict=False)
            self.assertTrue(all(k.startswith(prefix) for k in mismatch.missing_keys))
            self.assertEqual(mismatch.unexpected_keys, [])
            torch.testing.assert_close(
                model(inputs)[0], base(inputs if base.uses_market else inputs[:2])[0]
            )
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
            self.assertEqual(
                sum(p.numel() for p in model.market_gate.parameters()), 102
            )
            self.assertFalse(hasattr(model, "film"))
            self.assertFalse(hasattr(model, "market"))
            # Make market response nonzero before testing the next nested identity.
            torch.nn.init.normal_(model.market_gate.output.weight)
            base = model

    def test_independent_readiness_bounds_and_contribution(self):
        stock, perf, market = sample()
        model = FactorResidual([12, 12, 30], mode="contextual_v4_events", dropout=0)
        for gate in (model.market_gate, model.event_gate):
            with torch.no_grad():
                gate.project.weight.fill_(0.1)
                gate.output.weight.fill_(1)
        perf[..., 1::2] = 0
        _, out = model([stock, perf, market])
        self.assertFalse(out["performance_ready"].any())
        self.assertTrue(out["market_ready"])
        self.assertTrue(out["event_ready"])
        self.assertTrue(out["context_correction"].abs().sum() > 0)
        missing_market = market.clone()
        missing_market[..., 13::2] = 0
        _, out = model([stock, perf, missing_market])
        self.assertFalse(out["market_ready"])
        self.assertTrue(out["event_ready"])
        self.assertTrue(out["context_correction"].abs().sum() > 0)
        # All branches missing -> Temporal, irrespective of learned projections.
        missing_market[..., 1::2] = 0
        score, out = model([stock, perf, missing_market])
        torch.testing.assert_close(score, out["temporal_score"])
        self.assertEqual(out["loss_context_shrinkage"].item(), 0)
        # Market/event absence must leave a mature performance branch active.
        perf[..., 1::2] = 1
        with torch.no_grad():
            model.performance_gate.output.bias.fill_(1)
        _, out = model([stock, perf, missing_market])
        self.assertTrue(out["performance_ready"].all())
        self.assertTrue(out["context_correction"].abs().sum() > 0)
        # Sum before tanh: all branches together still obey a single +/-20% bound.
        for gate in (model.market_gate, model.event_gate):
            with torch.no_grad():
                gate.output.weight.fill_(1e4)
        score, out = model([stock, perf, market])
        torch.testing.assert_close(out["factor_gate"], torch.full((3,), 1.2))
        expected = torch.nn.functional.linear(
            stock[:, -1, :3] * 0.2, model.linear.weight
        )
        torch.testing.assert_close(out["context_correction"], expected)
        torch.testing.assert_close(score, out["linear_score"] + out["residual_score"])

    def test_market_model_ignores_events_and_contract_is_strict(self):
        inputs = list(sample())
        model = FactorResidual([12, 12, 30], mode="contextual_v4_market", dropout=0)
        torch.nn.init.normal_(model.market_gate.output.weight)
        before = model(inputs)[0]
        inputs[2][..., :12] = float("nan")
        torch.testing.assert_close(before, model(inputs)[0])
        torch.testing.assert_close(before, model([x.unsqueeze(2) for x in inputs])[0])
        for wrong in (
            torch.zeros(7, 1, 30),
            torch.zeros(1, 120, 30),
            torch.zeros(1, 1, 18),
        ):
            with self.assertRaises(ValueError):
                model([*inputs[:2], wrong])
        with self.assertRaises(ValueError):
            FactorResidual([12, 12, 18], mode="contextual_v4_market")

    def test_training_gradients_and_roundtrip(self):
        from src.res.model.util.core import BatchData, BatchInput, BatchOutput
        from src.res.model.util.metric.functions import LossFunction
        from src.res.model.util.metric.aggregator import LossAggregator

        for mode in ("contextual_v4_market", "contextual_v4_events"):
            inputs = sample()
            target = torch.randn(7, 1)
            data = BatchInput(
                list(inputs),
                target,
                None,
                torch.zeros(7, 2, dtype=torch.long),
                torch.ones(7, dtype=torch.bool),
                np.array([20170103]),
                np.arange(7),
            )
            model = FactorResidual([12, 12, 30], mode=mode, dropout=0)
            optim = torch.optim.Adam(model.parameters(), lr=0.001)
            for _ in range(4):
                optim.zero_grad()
                result = model(inputs)
                loss = LossFunction({"mse": {}}).losses(
                    BatchData(data, BatchOutput(result))
                )
                self.assertEqual(set(loss), {"mse", "loss_context_shrinkage"})
                LossAggregator().loss(loss).backward()
                optim.step()
            layers = [
                model.project,
                model.performance_gate.hidden,
                model.market_gate.project,
                model.market_gate.output,
            ]
            if mode == "contextual_v4_events":
                layers += [model.event_gate.project, model.event_gate.output]
            for layer in layers:
                assert layer.weight.grad is not None
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
            saved = BytesIO()
            torch.save(model.state_dict(), saved)
            saved.seek(0)
            clone = FactorResidual([12, 12, 30], mode=mode, dropout=0)
            clone.load_state_dict(torch.load(saved, weights_only=True))
            torch.testing.assert_close(model(inputs)[0], clone(inputs)[0])


if __name__ == "__main__":
    unittest.main()
