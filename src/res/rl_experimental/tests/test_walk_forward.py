import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.res.rl_experimental.data import make_synthetic_panel, Standardizer
from src.res.rl_experimental.env import PortfolioEnv, EpisodeRange
from src.res.rl_experimental.portfolio import PortfolioConstraints, project_action
from src.res.rl_experimental.baselines import replay_baseline
from src.res.rl_experimental.training import build_candidate_cache
from src.res.rl_experimental.walk_forward import (
    intervals, choose_screen, task_plan, load_config, cash_account, slice_panel, run_walk_forward,
)


def real_like_panel():
    dates = pd.bdate_range('2017-01-04', '2023-07-03')
    n = len(dates)-1
    p = make_synthetic_panel(n, 60, 3, seed=19)
    ds = dates.strftime('%Y%m%d').astype(int).to_numpy()
    returns = np.nan_to_num(p.forward_returns, nan=0.)
    mask = np.ones((n, 60), bool)
    # All assets eligible simplifies continuity checks without hiding execution constraints.
    return replace(p, decision_dates=ds[:-1], return_end_dates=ds[1:], execution_dates=ds[1:],
                   investable=mask, forward_returns=returns, overnight_returns=np.zeros_like(returns),
                   intraday_returns=returns, can_buy=mask, can_sell=mask,
                   valuation_stale=np.zeros_like(mask), risk_excluded=np.zeros_like(mask),
                   risk_reason=np.zeros(mask.shape, np.uint8), delisting_event=np.zeros_like(mask))


class WalkForwardTest(unittest.TestCase):
    def test_holding_bonus_and_risk_priority(self):
        constraints = PortfolioConstraints(candidate_count=3, max_holdings=1, max_weight=1, holding_bonus=.5)
        w, _, _ = project_action(np.zeros((3, 2)), np.array([.9, .8, .7]), np.ones(3, bool),
                                 np.arange(3), constraints, np.array([0, 1, 0]))
        self.assertEqual(w[1], 1)
        w, _, _ = project_action(np.zeros((3, 2)), np.array([.9, .8, .7]), np.array([1, 0, 1], bool),
                                 np.arange(3), constraints, np.array([0, 1, 0]))
        self.assertEqual(w[1], 0)

    def test_cost_decomposition_and_account_continuity(self):
        panel = slice_panel(real_like_panel(), 20230101, 20230201)
        c = PortfolioConstraints(extra_cost_bp=5)
        state = cash_account(panel)
        complete = replay_baseline(panel, (0, panel.n_steps), c, 'robust_top50')
        first = replay_baseline(panel, (0, 10), c, 'robust_top50', account_state=state)
        second = replay_baseline(panel, (10, panel.n_steps), c, 'robust_top50', account_state=state)
        np.testing.assert_allclose([r['nav'] for r in complete], [r['nav'] for r in first+second])
        for r in complete:
            self.assertAlmostEqual(r['fee_estimate'], r['base_fee']+r['extra_cost'], places=10)
        cheap = replay_baseline(panel, (0, panel.n_steps), replace(c, extra_cost_bp=0), 'robust_top50')
        self.assertLess(complete[-1]['nav'], cheap[-1]['nav'])

    def test_cached_observation_and_lean_history_equivalent(self):
        panel = slice_panel(real_like_panel(), 20230101, 20230201)
        slow = PortfolioEnv(panel, EpisodeRange(0, panel.n_steps), observation_version=2, cache_observation=False)
        fast = PortfolioEnv(panel, EpisodeRange(0, panel.n_steps), observation_version=2,
                            candidate_cache=build_candidate_cache(panel, 100), record_history=False)
        a, _ = slow.reset()
        b, _ = fast.reset()
        for i in range(panel.n_steps):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a['account'][1], 0)
            action = np.random.default_rng(i).normal(0, .1, slow.action_space.shape).astype(np.float32)
            a, r1, _, _, x = slow.step(action)
            b, r2, _, _, y = fast.step(action)
            self.assertEqual(r1, r2)
            self.assertEqual(x['nav'], y['nav'])
        self.assertFalse(fast.history)
        self.assertTrue(slow.history)

    def test_env_switch_preserves_state_and_horizon_independence(self):
        panel = slice_panel(real_like_panel(), 20230101, 20230201)
        full = PortfolioEnv(panel, EpisodeRange(0, panel.n_steps), observation_version=2)
        part = PortfolioEnv(panel, EpisodeRange(0, 10), observation_version=2)
        a, _ = full.reset()
        b, _ = part.reset()
        action = np.zeros(full.action_space.shape, np.float32)
        for _ in range(10):
            full.step(action)
            part.step(action)
        tail = PortfolioEnv(panel, EpisodeRange(10, panel.n_steps), observation_version=2)
        tail.reset(options={'account_state': part.export_account()})
        for _ in range(10, panel.n_steps):
            a, _, _, _, x = full.step(action)
            b, _, _, _, y = tail.step(action)
            self.assertEqual(x['nav'], y['nav'])
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_intervals_and_training_only_normalization(self):
        p = real_like_panel()
        splits = intervals(p, 20180101, 20220101, 20230101, 20230401)
        self.assertTrue(np.all(p.return_end_dates[slice(*splits['train'])] < 20220101))
        self.assertTrue(np.all(p.return_end_dates[slice(*splits['valid'])] < 20230101))
        future = replace(p, features=p.features.copy())
        future.features[splits['valid'][0]:] += 100
        a = Standardizer.fit(p, *splits['train'])
        b = Standardizer.fit(future, *splits['train'])
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.scale, b.scale)

    def test_screen_selection_rule(self):
        rows = [{'penalty': p, 'bonus': 0., 'seed': seed, 'turnover_ratio': t, 'relative_nav': n}
                for p, t, n in [(.0003, 2, .2), (.001, 1.3, .1), (.003, 1.2, .05)] for seed in (7, 17, 29)]
        self.assertEqual(choose_screen(rows, False)['penalty'], .001)
        for r in rows:
            r['turnover_ratio'] *= 2
        self.assertEqual(choose_screen(rows, False)['penalty'], .003)

    def test_default_plan(self):
        c = load_config(Path(__file__).parents[1]/'example_walk_forward.json')
        p = task_plan(c)
        self.assertEqual(p['total_training_tasks'], 132)
        self.assertEqual(p['folds'][0]['start'], 20180101)
        self.assertEqual(p['folds'][-1]['end'], 20260101)

    def test_small_suite_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            real_like_panel().save_npz(path/'panel.npz')
            c = dict(panel=str(path/'panel.npz'), output=str(path/'run'),
                     screen=dict(start=20170101, valid_start=20210101, end=20230101),
                     first_quarter='2023Q1', last_quarter='2023Q2', seeds=[7], penalties=[.001], bonuses=[0, .5],
                     costs_bp=[3, 5], screen_steps=16, quarter_steps=16,
                     training=dict(device='cpu', n_envs=1, rollout_steps=8, batch_size=8, update_epochs=1, hidden_dim=8))
            (path/'config.json').write_text(json.dumps(c))
            out = run_walk_forward(path/'config.json')
            self.assertEqual(out['status'], 'success')
            report = json.loads((path/'run/summary/summary.json').read_text())
            self.assertIn('retention', report['judgments'])
            self.assertEqual(len(report['quarter_returns']['robust/3']), 2)  # No partial July quarter
            model = path/'run/quarters/2023Q1/penalty/seed-7/best_model.zip'
            stamp = model.stat().st_mtime_ns
            run_walk_forward(path/'config.json', stage='screen')
            self.assertEqual(model.stat().st_mtime_ns, stamp)
            (path/'run/selected.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'modified'):
                run_walk_forward(path/'config.json', stage='quarters')
