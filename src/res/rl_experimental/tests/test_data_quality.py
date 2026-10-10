import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.res.rl_experimental.data_quality import deduplicate_table
from src.res.rl_experimental.real_data import _load_table


class DataQualityTest(unittest.TestCase):
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
