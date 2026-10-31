import tempfile
import unittest
from typing import Any
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.res.rl_experimental.data import PanelData
from src.res.rl_experimental.portfolio import execute_open_close, PortfolioConstraints
from src.res.rl_experimental.risk import delisting_events, prepare_risk, warning_ids
from src.res.rl_experimental.env import EpisodeRange, PortfolioEnv, zero_action
from src.res.rl_experimental.baselines import replay_baseline, history_metrics


class DelistingRiskTest(unittest.TestCase):
    def execute(self, old, cash, events, target=None):
        n = len(old)
        return execute_open_close(np.array(old), cash, np.zeros(n) if target is None else np.array(target),
                                  np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool),
                                  fee_rate=.00035, max_holdings=50, max_weight=1.,
                                  stock_ids=np.arange(n) + 540, execution_date=20230630,
                                  delisting_event=np.array(events))

    def test_000540_and_multiple_writeoffs(self):
        result = self.execute([.014098115265369415], 1-.014098115265369415, [True])
        self.assertAlmostEqual(result.nav_multiplier, 1-.014098115265369415)
        self.assertEqual(result.end_weights[0], 0)
        self.assertEqual(result.cost_fraction, 0)
        self.assertEqual(result.turnover, 0)
        self.assertEqual(result.end_cash, 1)
        result = self.execute([.2, .3], .5, [True, True])
        self.assertEqual(result.nav_multiplier, .5)
        self.assertEqual(result.written_off_indices, (0, 1))
        again = self.execute([0, 0], 1, [True, True])
        self.assertEqual(again.delisting_loss_fraction, 0)
        self.assertEqual(again.nav_multiplier, 1)
        with self.assertRaisesRegex(ValueError, 'overnight'):
            self.execute([1], 0, [False])  # missing != delisted

    def panel(self):
        dates = np.array([20230628, 20230629, 20230630])
        ends = np.array([20230629, 20230630, 20230703])
        return PanelData(stock_ids=np.array([540]), decision_dates=dates, return_end_dates=ends,
                         features=np.zeros((3, 1, 1)), alpha=np.ones((3, 1)), investable=np.array([[1], [0], [0]], bool),
                         forward_returns=np.zeros((3, 1)), execution_dates=ends,
                         overnight_returns=np.array([[0], [np.nan], [np.nan]]),
                         intraday_returns=np.array([[0], [np.nan], [np.nan]]),
                         can_buy=np.array([[1], [0], [0]], bool), can_sell=np.array([[1], [0], [0]], bool),
                         valuation_stale=np.zeros((3, 1), bool), risk_excluded=np.array([[0], [1], [1]], bool),
                         risk_reason=np.array([[0], [1], [1]], np.uint8), delisting_event=np.array([[0], [1], [0]], bool))

    def test_bankruptcy_shared_env_baselines_and_contract(self):
        panel = self.panel()
        constraints = PortfolioConstraints(candidate_count=1, max_holdings=50, max_weight=1, fee_rate=0)
        for kind in ('alpha_equal_weight', 'robust_top50'):
            history = replay_baseline(panel, (0, 3), constraints, kind)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[-1]['nav'], 0)
            self.assertEqual(history[-1]['period_return'], -1)
            self.assertEqual(history[-1]['written_off_stock_ids'], [540])
            self.assertEqual(history_metrics(history)['total_delisting_loss'], 1)
        env = PortfolioEnv(panel, EpisodeRange(0, 3), constraints, random_start=False)
        env.reset()
        env.step(zero_action(env.action_space))
        _, reward, done, _, info = env.step(zero_action(env.action_space))
        self.assertTrue(done)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info['nav'], 0)
        self.assertEqual(info['period_return'], -1)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            env.step(zero_action(env.action_space))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'panel.npz'
            panel.save_npz(path)
            self.assertEqual(panel.contract(), PanelData.load_npz(path).contract())
            changed = replace(panel, delisting_event=np.zeros((3, 1), bool))
            self.assertNotEqual(panel.contract(), changed.contract())
            other = PortfolioEnv(changed, EpisodeRange(0, 3), constraints, random_start=False)
            first = PortfolioEnv(panel, EpisodeRange(0, 3), constraints, random_start=False)
            a, _ = first.reset()
            b, _ = other.reset()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_risk_freeze_then_writeoff(self):
        panel = self.panel()
        panel = replace(panel, delisting_event=np.array([[0], [0], [1]], bool),
                        overnight_returns=np.array([[0], [0], [np.nan]]),
                        intraday_returns=np.array([[0], [0], [np.nan]]))
        constraints = PortfolioConstraints(candidate_count=1, max_holdings=50, max_weight=1, fee_rate=0)
        for kind in ('alpha_equal_weight', 'robust_top50'):
            h = replay_baseline(panel, (0, 3), constraints, kind)
            self.assertEqual(h[1]['risk_unsold_holdings'], 1)
            self.assertEqual(h[1]['rejected_sells'], 1)
            self.assertEqual(h[1]['turnover'], 0)
            self.assertEqual(h[2]['nav'], 0)

    def test_calendar_and_warning_window(self):
        desc = pd.DataFrame({'secid': [540], 'delist_dt': [20230701]})
        events = delisting_events(np.array([20230629, 20230630, 20230703]),
                                  np.array([20230630, 20230703, 20230704]), [540], desc)
        np.testing.assert_array_equal(events[:, 0], [False, True, False])
        frame = pd.DataFrame({'secid': [1, 2, 3, 4, 5], 'date': [20230402, 20230401, 20230630, 20230701, 20230630],
                              'title': ['可能被风险警示', '可能终止退市', '可能退市', '可能退市', '无风险']})
        np.testing.assert_array_equal(warning_ids(frame, 20230630), [1, 3])

    def test_risk_sale_and_recovery(self):
        args: dict[str, Any] = dict(old_weights=np.array([1.]), old_cash=0., target_weights=np.array([1.]),
                    overnight_returns=np.array([0.]), intraday_returns=np.array([0.]),
                    can_buy=np.array([True]), can_sell=np.array([True]),
                    fee_rate=.001, max_holdings=50, max_weight=1.)
        sold = execute_open_close(**args, risk_excluded=np.array([True]))
        self.assertEqual(sold.end_weights[0], 0)
        self.assertAlmostEqual(sold.nav_multiplier, .999)
        self.assertEqual(sold.turnover, 1)
        args.update(old_weights=np.array([0.]), old_cash=1.)
        held = execute_open_close(**args, risk_excluded=np.array([False]))
        self.assertGreater(held.end_weights[0], 0)

    def test_cache_and_available_announcements_are_causal(self):
        class DB:
            def dates(self, source, key):
                return [20230629, 20230630] if key == 'announcement_szse' else []
            def path(self, source, key, date):
                return SimpleNamespace(exists=lambda: date == 20230629)
            def load(self, source, key, date, **kwargs):
                if source == 'exposure':
                    return pd.DataFrame({'secid': [1, 9], 'date': [20230629, 20230701]})
                return pd.DataFrame({'secid': [2, 3], 'date': [date, 20230701],
                                     'title': ['可能退市', '可能风险警示']})
        info = SimpleNamespace(get_st=lambda date: pd.DataFrame({'secid': []}))
        masks, _, report = prepare_risk(np.array([20230629, 20230630]), np.array([1, 2, 3, 9]), DB(), info)
        np.testing.assert_array_equal(masks, [[1, 0, 0, 0], [0, 1, 0, 0]])
        self.assertEqual(report['coverage'][0]['source'], 'daily_cache')

    def test_zero_benchmark_comparison(self):
        from src.res.rl_experimental.diagnostics import compare_metrics, comparison_figure
        panel = self.panel()
        history = replay_baseline(panel, (0, 3), PortfolioConstraints(max_weight=1., fee_rate=0), 'alpha_equal_weight')
        metrics = history_metrics(history)
        self.assertIsNone(compare_metrics(metrics, metrics)['relative_nav_return'])
        longer = history + [dict(history[-1], decision_date=20230630, return_end_date=20230703)]
        figure = comparison_figure({'ppo': history, 'equal_weight': longer})
        self.assertEqual(len(figure.axes), 6)

    def test_read_only_missing_warning_data(self):
        class DB:
            def dates(self, *args): return []
            def path(self, *args): return Path('/does-not-exist/rl-risk')
            def load(self, *args): raise AssertionError('unexpected read')
        info = SimpleNamespace(get_st=lambda date: pd.DataFrame({'secid': [540]}))
        excluded, reasons, report = prepare_risk(np.array([20230629]), np.array([540, 1]), DB(), info)
        np.testing.assert_array_equal(excluded, [[True, False]])
        np.testing.assert_array_equal(reasons, [[1, 0]])
        self.assertEqual(report['coverage'][0]['missing_exchange_days'], 270)

    def test_empty_announcements_are_quiet_but_weekend_risks_are_preserved(self):
        calls = []
        class DB:
            def dates(self, *args): return [20230624, 20230625, 20230626]
            def path(self, *args): return Path('/does-not-exist/rl-risk')
            def load(self, source, key, date, **kwargs):
                calls.append((key, date, kwargs))
                if key == 'announcement_szse' and date == 20230624:
                    return pd.DataFrame({'secid': [2], 'date': [date], 'title': ['可能退市']})
                return pd.DataFrame()
        info = SimpleNamespace(get_st=lambda date: pd.DataFrame({'secid': []}))
        with patch('src.res.rl_experimental.risk.progress') as progress:
            excluded, _, report = prepare_risk(np.array([20230626]), np.array([1, 2]), DB(), info)
        np.testing.assert_array_equal(excluded, [[False, True]])
        self.assertEqual(len(calls), 9)
        self.assertTrue(all(c[2] == {'vb_level': 'never'} for c in calls))
        self.assertEqual(report['announcement_file_counts']['bse'],
                         {'loaded': 3, 'empty': 3, 'empty_weekend': 2})
        self.assertEqual(report['announcement_file_counts']['szse']['empty'], 2)
        self.assertEqual(report['coverage'][0]['coverage'], 'incomplete')
        self.assertTrue(progress.call_args.kwargs['warning'])


if __name__ == '__main__':
    unittest.main()
