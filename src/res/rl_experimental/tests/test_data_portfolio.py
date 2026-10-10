from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.res.rl_experimental.data import PanelData, Standardizer, chronological_split, make_synthetic_panel
from src.res.rl_experimental.portfolio import PortfolioConstraints, execute_open_close, project_action, settle_period


class DataAndPortfolioTest(unittest.TestCase):
    def test_npz_round_trip_and_train_only_standardization(self) -> None:
        panel = make_synthetic_panel(30, 12, 3, seed=11)
        splits = chronological_split(panel.n_steps)
        standardizer = Standardizer.fit(panel, *splits["train"])
        normalized = standardizer.transform(panel)
        training_values = normalized.features[: splits["train"][1]][panel.investable[: splits["train"][1]]]
        np.testing.assert_allclose(training_values.mean(axis=0), 0.0, atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.npz"
            panel.save_npz(path)
            loaded = PanelData.load_npz(path)
        np.testing.assert_array_equal(loaded.stock_ids, panel.stock_ids)
        np.testing.assert_allclose(loaded.features, panel.features)

    def test_execution_panel_round_trip_preserves_schema(self) -> None:
        panel = make_synthetic_panel(12, 8, 3, seed=19)
        shape = panel.alpha.shape
        real = replace(
            panel,
            execution_dates=panel.return_end_dates.copy(),
            overnight_returns=np.zeros(shape, dtype=np.float32),
            intraday_returns=np.nan_to_num(panel.forward_returns, nan=0.0),
            can_buy=panel.investable.copy(),
            can_sell=np.ones(shape, dtype=bool),
            valuation_stale=np.zeros(shape, dtype=bool),
            industry_codes=np.tile(np.arange(8, dtype=np.int16) % 3, (12, 1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real.npz"
            real.save_npz(path)
            loaded = PanelData.load_npz(path)
        self.assertTrue(loaded.is_execution_panel)
        self.assertEqual(loaded.observation_feature_count, real.n_features + 3)
        self.assertEqual(loaded.contract(), real.contract())
        np.testing.assert_array_equal(loaded.can_buy, real.can_buy)

    def test_project_action_obeys_hard_constraints(self) -> None:
        constraints = PortfolioConstraints(candidate_count=6, max_holdings=3, max_weight=0.25)
        action = np.zeros((8, 2), dtype=np.float32)
        alpha = np.arange(8, dtype=np.float32)
        mask = np.array([1, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
        ids = np.arange(100, 108)
        weights, cash, chosen = project_action(action, alpha, mask, ids, constraints)
        self.assertLessEqual(np.count_nonzero(weights), 3)
        self.assertLessEqual(float(weights.max()), 0.25 + 1e-7)
        self.assertTrue(np.all(weights[~mask] == 0))
        self.assertAlmostEqual(float(weights.sum()) + cash, 1.0, places=6)
        self.assertEqual(len(chosen), 3)

    def test_transaction_cost_identity_and_weight_drift(self) -> None:
        old = np.array([0.5, 0.0], dtype=np.float32)
        target = np.array([0.0, 0.8], dtype=np.float32)
        end_weights, end_cash, nav, turnover, cost = settle_period(
            old, target, 0.2, np.array([0.0, 0.1]), fee_rate=0.001
        )
        self.assertAlmostEqual(cost, 0.001 * turnover, places=10)
        self.assertAlmostEqual(float(end_weights.sum()) + end_cash, 1.0, places=6)
        self.assertGreater(end_weights[1], 0.8)
        self.assertGreater(nav, 1.0)

    def test_open_execution_freezes_untradeable_position(self) -> None:
        result = execute_open_close(
            old_weights=np.array([0.6, 0.0]),
            old_cash=0.4,
            target_weights=np.array([0.0, 0.6]),
            overnight_returns=np.array([0.0, 0.0]),
            intraday_returns=np.array([0.0, 0.0]),
            can_buy=np.array([False, True]),
            can_sell=np.array([False, True]),
            fee_rate=0.001,
            max_holdings=1,
            max_weight=0.6,
        )
        self.assertAlmostEqual(result.end_weights[0], 0.6, places=7)
        self.assertEqual(result.rejected_sell_count, 1)
        self.assertEqual(result.rejected_buy_count, 1)
        self.assertEqual(result.frozen_count, 1)
        self.assertAlmostEqual(result.turnover, 0.0)

    def test_open_execution_sells_before_cash_scaled_buys(self) -> None:
        result = execute_open_close(
            old_weights=np.array([0.8, 0.0]),
            old_cash=0.2,
            target_weights=np.array([0.0, 0.8]),
            overnight_returns=np.array([0.0, 0.0]),
            intraday_returns=np.array([0.0, 0.1]),
            can_buy=np.array([True, True]),
            can_sell=np.array([True, True]),
            fee_rate=0.001,
            max_holdings=1,
            max_weight=0.8,
        )
        self.assertAlmostEqual(result.sell_notional, 0.8, places=8)
        self.assertGreater(result.buy_notional, 0.79)
        self.assertAlmostEqual(result.cost_fraction, 0.001 * result.turnover, places=10)
        self.assertEqual(np.count_nonzero(result.end_weights > 1e-8), 1)
        self.assertGreater(result.nav_multiplier, 1.07)


if __name__ == "__main__":
    unittest.main()
