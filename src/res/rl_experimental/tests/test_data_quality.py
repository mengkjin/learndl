import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from typing import Any

import numpy as np
import pandas as pd

from src.res.rl_experimental.data_quality import deduplicate_table
from src.res.rl_experimental.real_data import _execution_returns, _load_table
from src.res.rl_experimental.portfolio import execute_open_close


class DataQualityTest(unittest.TestCase):
    def test_missing_close_does_not_invalidate_known_open(self):
        overnight, intraday = _execution_returns(
            np.array([10., 10., 10.]), np.array([np.nan, 10., np.nan]),
            np.array([11., np.nan, np.nan]), np.array([True, False, False]),
        )
        self.assertAlmostEqual(overnight[0], .1)
        self.assertTrue(np.isnan(intraday[0]))
        self.assertEqual(overnight[1], 0.)
        self.assertEqual(overnight[2], 0.)
        self.assertTrue(np.isnan(intraday[2]))
        # Selling all at a known open needs no subsequent close valuation.
        args: dict[str, Any] = dict(old_weights=np.array([.5]), old_cash=.5,
                    target_weights=np.array([0.]), overnight_returns=overnight[:1],
                    intraday_returns=intraday[:1], can_buy=np.array([True]), can_sell=np.array([True]),
                    fee_rate=0., max_holdings=1, max_weight=1., stock_ids=np.array([600001]),
                    decision_date=20230103, execution_date=20230104)
        self.assertAlmostEqual(execute_open_close(**args).nav_multiplier, 1.05)
        args['can_sell'] = np.array([False])
        with self.assertRaisesRegex(ValueError, 'a close valuation.*execution_date=20230104.*600001'):
            execute_open_close(**args)
        args['overnight_returns'] = np.array([np.nan])
        with self.assertRaisesRegex(ValueError, 'an overnight valuation.*decision_date=20230103.*600001'):
            execute_open_close(**args)

    def test_duplicates_warn_keep_last_and_leave_source_untouched(self):
        frame = pd.DataFrame({'date': [1, 1, 2, 2], 'secid': [3, 3, 3, 3], 'value': [4., 4., 5., 6.]})
        original = frame.copy(deep=True)
        audit = []
        errors = io.StringIO()
        with redirect_stderr(errors):
            clean = deduplicate_table(frame, ['date', 'secid'], 'source', audit)
        self.assertEqual(clean['value'].tolist(), [4., 6.])
        pd.testing.assert_frame_equal(frame, original)
        self.assertIn('WARNING', errors.getvalue())
        self.assertIn('conflicting keys=1', errors.getvalue())
        self.assertEqual(audit[0]['removed_rows'], 2)

    def test_unique_rows_are_unchanged_and_silent(self):
        frame = pd.DataFrame({'secid': [1, 2], 'value': [1., 2.]})
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertIs(deduplicate_table(frame, ['secid'], 'source'), frame)
        self.assertEqual(errors.getvalue(), '')

    def test_batched_loading_reports_progress_and_deduplicates(self):
        def load(source, key, dates, **kwargs):
            frame = pd.DataFrame({'date': dates, 'secid': 1, 'value': 2.})
            return pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        audit = []
        output, errors = io.StringIO(), io.StringIO()
        with patch('src.res.rl_experimental.real_data.DB.loads', side_effect=load) as reader, redirect_stdout(output), redirect_stderr(errors):
            frame = _load_table('trade_ts', 'day_limit', np.arange(257), 'date', audit)
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(len(frame), 257)
        self.assertEqual(audit[0]['removed_rows'], 3)
        self.assertIn('257-257/257', output.getvalue())


if __name__ == '__main__':
    unittest.main()
