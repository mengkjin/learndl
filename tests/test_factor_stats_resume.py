"""Resume must rebuild missing current-version group statistics from predictions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch
from typing import Any

import numpy as np
import pandas as pd

from src.proj import Const
from src.res.factor.analytic.factor_perf import FactorPerfTest
from src.res.factor.util.classes.stock_factor import CacheFactorStats, FactorStats, StockFactor
from src.res.model.callback.test.detail import DetailedAlphaAnalysis

__all__ = ['FactorStatsResumeTest']


class FactorStatsResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'factor_perf_test' / 'factor_stats'
        self.dates = np.array([20260105, 20260112, 20260119])
        self.required = FactorPerfTest.required_group_stats()
        for name in ('ic', 'ic_indus'):
            self.write(name, 'bm=market', self.dates)

    def write(self, name, key, dates):
        folder = self.path / name
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'date': dates}).to_feather(folder / f'{key}.feather')

    def groups(self, dates=None, old=False, skip=None):
        for i, params in enumerate(self.required):
            if i == skip:
                continue
            params = dict(params)
            if old:
                params.pop('grouping_version')
            self.write('group_perf', FactorStats.param_to_str(params),
                       self.dates if dates is None else dates)

    def saved(self):
        return FactorPerfTest.factor_stats_saved_dates(self.root)

    def test_unversioned_history_is_not_complete(self):
        self.groups(old=True)
        self.assertEqual(self.saved().size, 0)

    def test_wrong_version_is_not_complete(self):
        for params in self.required:
            self.write('group_perf', FactorStats.param_to_str(params | {'grouping_version': 'old'}), self.dates)
        self.assertEqual(self.saved().size, 0)

    def test_absent_group_directory_is_not_complete(self):
        self.assertEqual(self.saved().size, 0)

    def test_each_required_variant_must_exist(self):
        for i, params in enumerate(self.required):
            if params['bm'] == 'csi2000':
                continue
            with self.subTest(params=params):
                self.groups()
                (self.path / 'group_perf' / f'{FactorStats.param_to_str(params)}.feather').unlink()
                self.assertEqual(self.saved().size, 0)

    def test_partial_current_coverage_and_stale_files(self):
        self.groups(old=True)
        self.groups(self.dates[1:])
        np.testing.assert_array_equal(self.saved(), self.dates[1:])
        self.groups()
        self.groups(self.dates[:1], old=True)
        np.testing.assert_array_equal(self.saved(), self.dates)

    def test_one_current_variant_has_a_history_hole(self):
        self.groups()
        params = next(p for p in self.required if p.get('rank') and p['bm'] != 'csi2000')
        self.write('group_perf', FactorStats.param_to_str(params), self.dates[[0, 2]])
        np.testing.assert_array_equal(self.saved(), self.dates[[0, 2]])

    def test_empty_current_variant_is_not_complete(self):
        self.groups()
        self.write('group_perf', FactorStats.param_to_str(self.required[0]), [])
        self.assertEqual(self.saved().size, 0)

    def test_direct_cache_check_rejects_old_version(self):
        self.groups(old=True)
        self.assertEqual(CacheFactorStats.saved_dates(self.path).size, 0)

    def test_csi2000_exclusion_is_preserved(self):
        self.groups()
        self.write('group_perf', FactorStats.param_to_str(FactorStats.group_perf_params('csi2000')), [])
        np.testing.assert_array_equal(self.saved(), self.dates)

    def test_resume_loads_missing_history_and_rebuilds_without_training(self):
        self.groups(old=True)
        self.groups(self.dates[1:])
        loaded = []

        def get_factor(dates):
            loaded.extend(dates.tolist())
            frame = pd.DataFrame([(date, secid, float(secid)) for date in dates for secid in range(100)],
                                 columns=['date', 'secid', 'best'])
            return StockFactor(frame, factor_names=['best'])

        # The callback samples every fifth trading day before comparing coverage.
        callback: Any = SimpleNamespace(test_dates=np.repeat(self.dates, 5), snap_folder=self.root,
                                   get_factor=get_factor)
        with patch.object(type(Const.Model), 'resume_factor_perf', new_callable=PropertyMock, return_value=True):
            factor = DetailedAlphaAnalysis.get_factor_for_factor_test(callback)
        self.assertEqual(loaded, [int(self.dates[0])])
        np.testing.assert_array_equal(factor.date, self.dates)
        cache = CacheFactorStats()
        # Seed current cached results, then exercise the real group computation.
        for params in self.required:
            cache.group_perf.update_stat(params, pd.DataFrame({
                'date': self.dates[1:], 'factor_name': 'best', 'group': 1,
                'group_ret': .01, 'start': self.dates[1:], 'end': self.dates[1:]}))
        factor.cache_factor_stats = cache
        frame = factor.frame().assign(ret=lambda df: df['best'] / 1000)
        with patch.object(factor, 'frame_with_cols', return_value=frame) as inputs:
            for params in self.required:
                factor.benchmark = params['bm']
                result = factor.eval_group_perf(params['n'], params['lag'], params['gp'],
                                                params['exc'], params['ret'], as_rank=params.get('rank', False))
                np.testing.assert_array_equal(np.sort(result.date.unique()), self.dates)
            for call in inputs.call_args_list:
                np.testing.assert_array_equal(call.kwargs['dates'], self.dates[:1])
        cache.save(self.path)
        np.testing.assert_array_equal(self.saved(), self.dates)
        callback.get_factor = Mock(side_effect=get_factor)
        with patch.object(type(Const.Model), 'resume_factor_perf', new_callable=PropertyMock, return_value=True):
            DetailedAlphaAnalysis.get_factor_for_factor_test(callback)
        np.testing.assert_array_equal(callback.get_factor.call_args.args[0], self.dates[-1:])
