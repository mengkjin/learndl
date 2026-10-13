import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.res.rl_experimental.valuation import build_valuations
from src.res.rl_experimental.real_data import _execution_returns
from src.res.rl_experimental.data import PanelData
from src.res.rl_experimental.portfolio import PortfolioConstraints, execute_open_close
from src.res.rl_experimental.baselines import replay_baseline
from src.res.rl_experimental.env import PortfolioEnv, EpisodeRange
from src.res.rl_experimental.experiment import _snapshot


class ValuationTest(unittest.TestCase):
    def sample(self):
        dates = np.arange(20250101, 20250108)
        raw = np.array([100., 100., np.nan, np.nan, 60., 66., np.nan])[:, None]
        factors = np.array([1., 1., np.nan, np.nan, 2., 2., np.nan])[:, None]
        status = np.array([1., 1., np.nan, np.nan, 1., 1., np.nan])[:, None]
        adj = np.array([100., 100., 100., np.nan, 120., 132., np.nan])[:, None]
        alive = np.ones_like(raw, dtype=bool)
        return dates, np.array([600001]), raw, factors, status, adj, alive

    def test_carry_resume_split_adjustment_and_causal_history(self):
        args = self.sample()
        values = build_valuations(*args)
        np.testing.assert_allclose(values.close[:, 0], [100, 100, 100, 100, 120, 132, 132])
        np.testing.assert_array_equal(values.reason[:, 0], [1, 1, 3, 4, 1, 1, 4])
        np.testing.assert_array_equal(values.stale_days[:, 0], [0, 0, 1, 2, 0, 0, 1])
        modified = list(args)
        modified[2] = args[2].copy()
        modified[2][4:] *= 2
        later = build_valuations(*modified)
        np.testing.assert_array_equal(values.close[:4], later.close[:4])
        np.testing.assert_array_equal(values.reason[:4], later.reason[:4])
        np.testing.assert_array_equal(values.stale_days[:4], later.stale_days[:4])
        opened = np.array([100., np.nan, np.nan, 110., 120., np.nan])[:, None]
        valid = np.isfinite(opened)
        overnight, intraday = _execution_returns(values.close[:-1], values.close[1:], opened, valid)
        self.assertAlmostEqual(overnight[3, 0], .1)
        nav = 1.
        for i in range(6):
            settled = execute_open_close(np.array([1.]), 0., np.array([1.]), overnight[i], intraday[i],
                                         valid[i], valid[i], fee_rate=0., max_holdings=1, max_weight=1.)
            nav *= settled.nav_multiplier
            if not valid[i, 0]:
                self.assertEqual(settled.turnover, 0.)
        self.assertAlmostEqual(nav, 1.32)

    def test_delisting_unknown_anchor_conflict_and_nontrade_mark(self):
        args = list(self.sample())
        args[6][-1] = False
        values = build_valuations(*args)
        self.assertTrue(np.isnan(values.close[-1, 0]))
        self.assertEqual(values.reason[-1, 0], 5)
        self.assertEqual(values.last_valid_date[-1, 0], 20250106)
        args[2][:2] = np.nan
        args[5][:2] = np.nan
        self.assertTrue(np.isnan(build_valuations(*args).close[0, 0]))
        args = list(self.sample())
        args[5][0] = 99.
        self.assertEqual(build_valuations(*args).report['conflicting_price_cells'], 1)
        overnight, intraday = _execution_returns(np.array([100.]), np.array([110.]),
                                                np.array([np.nan]), np.array([False]))
        self.assertEqual(overnight[0], 0.)
        self.assertAlmostEqual(intraday[0], .1)
        overnight, intraday = _execution_returns(np.array([100.]), np.array([80.]),
                                                np.array([90.]), np.array([True]))
        self.assertAlmostEqual((1 + overnight[0]) * (1 + intraday[0]), .8)

    def test_old_snapshot_requires_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'panel.npz').touch()
            (path / 'quality_report.json').write_text('{}')
            (path / 'manifest.json').write_text(json.dumps({'panel_contract': {'schema_version': 2}}))
            with self.assertRaisesRegex(ValueError, 'rebuild_snapshot=true'):
                _snapshot({'snapshot_dir': str(path), 'alpha': 'pred@test', 'start': 20250101, 'end': 20250201})

    def test_env_baseline_and_version_roundtrip(self):
        args = self.sample()
        values = build_valuations(*args)
        opened = np.array([100., np.nan, np.nan, 110., 120., np.nan])[:, None]
        valid = np.isfinite(opened)
        overnight, intraday = _execution_returns(values.close[:-1], values.close[1:], opened, valid)
        panel = PanelData(stock_ids=args[1], decision_dates=args[0][:-1], return_end_dates=args[0][1:],
                          execution_dates=args[0][1:], features=np.zeros((6, 1, 1)), alpha=np.ones((6, 1)),
                          investable=(args[4][:-1] > 0), forward_returns=values.close[1:] / values.close[:-1] - 1,
                          overnight_returns=overnight, intraday_returns=intraday, can_buy=valid, can_sell=valid,
                          valuation_stale=values.stale[1:], valuation_reason=values.reason[1:],
                          valuation_stale_days=values.stale_days[1:], valuation_last_valid_date=values.last_valid_date[1:])
        constraints = PortfolioConstraints(candidate_count=1, max_holdings=1, max_weight=1., fee_rate=0.)
        baseline = replay_baseline(panel, (0, 6), constraints, 'alpha_equal_weight')
        env = PortfolioEnv(panel, EpisodeRange(0, 6), constraints, random_start=False)
        env.reset()
        for _ in range(6):
            env.step(np.zeros(env.action_space.shape))
        np.testing.assert_allclose([r['nav'] for r in baseline], [r['nav'] for r in env.history])
        self.assertEqual(baseline[2]['missing_data_carried_holdings'], 1)
        self.assertEqual(baseline[2]['max_held_stale_days'], 2)
        # Do not silently continue a held position after its known listing ends.
        stopped = replace(panel, overnight_returns=overnight.copy(), intraday_returns=intraday.copy(),
                          can_buy=valid.copy(), can_sell=valid.copy(), valuation_reason=panel.valuation_reason.copy())
        stopped.overnight_returns[-1] = np.nan
        stopped.intraday_returns[-1] = np.nan
        stopped.can_buy[-1] = False
        stopped.can_sell[-1] = False
        stopped.valuation_reason[-1] = 5
        with self.assertRaisesRegex(ValueError, '600001.*not_listed'):
            replay_baseline(stopped, (0, 6), constraints, 'alpha_equal_weight')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'panel.npz'
            panel.save_npz(path)
            self.assertEqual(PanelData.load_npz(path).contract(), panel.contract())
            old = replace(panel, schema_version=2, valuation_reason=None,
                          valuation_stale_days=None, valuation_last_valid_date=None)
            old.save_npz(path)
            self.assertEqual(PanelData.load_npz(path).contract(), old.contract())


if __name__ == '__main__':
    unittest.main()
