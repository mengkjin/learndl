"""All-missing factor cross-sections must not crash normalization or first stats."""
import unittest

import numpy as np
import pandas as pd

from src.res.factor.util.classes.stock_factor import (
    StockFactor,
    _pivot_named_stat,
    normalize_df,
)


def _indexed(values: list[float], name: str = 'hy_scores_v5') -> pd.DataFrame:
    dates = np.repeat([20260105, 20260106], len(values) // 2)
    secids = np.tile(np.arange(1, len(values) // 2 + 1), 2)
    return pd.DataFrame(
        {name: values},
        index=pd.MultiIndex.from_arrays([dates, secids], names=['date', 'secid']),
    )


class FactorNormalizeEmptyTest(unittest.TestCase):
    def test_all_nan_cross_section_returns_empty_schema(self):
        df = _indexed([np.nan] * 4)
        out = normalize_df(df, fill_method='drop')
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ['date', 'secid', 'hy_scores_v5'])

    def test_partial_nan_still_normalizes(self):
        df = _indexed([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0])
        out = normalize_df(df, fill_method='drop')
        self.assertEqual(len(out), 4)
        self.assertTrue(np.isfinite(out['hy_scores_v5']).all())

    def test_first_stats_on_all_nan_factor_is_date_schema(self):
        raw = pd.DataFrame({
            'secid': [1, 2],
            'date': [20260105, 20260105],
            'hy_scores_v5': [np.nan, np.nan],
        })
        factor = StockFactor(raw)
        factor.normalize(fill_method='drop', inplace=True)
        stats = factor.weekly_stats()
        self.assertEqual(list(stats.columns), ['date'])
        saved = pd.concat([pd.DataFrame(), stats]).drop_duplicates(subset=['date'], keep='last')
        saved = saved.sort_values('date').reset_index(drop=True)
        self.assertTrue(saved.empty)
        self.assertIn('date', saved.columns)

    def test_named_stat_pivot_keeps_populated_groups(self):
        long = pd.DataFrame({
            'date': [20260105, 20260105],
            'group': [1, 2],
            'group_ret': [0.01, -0.02],
        })
        wide = _pivot_named_stat(long, value='group_ret', column='group', prefix='group@')
        self.assertEqual(list(wide.columns), ['group@1', 'group@2'])
        self.assertEqual(wide.index.names, ['date'])


if __name__ == '__main__':
    unittest.main()
