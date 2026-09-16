"""Backfill limits must support CLI null values without disabling stop guards."""
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.download.other_source.rcquant import bar_min


class BackfillLimitTest(unittest.TestCase):
    def run_backfill(self, kwargs, *, deadline=False):
        updater = MagicMock()
        updater.rcquant_bar_min.return_value = True
        with patch.object(bar_min, 'MACHINE', updatable=True, belong_to_hfm=False), \
             patch.object(bar_min, 'backfill_sec_dates', return_value=pd.Index(range(1, 8))), \
             patch.object(bar_min, '_backfill_deadline_hit', return_value=deadline), \
             patch.object(bar_min, 'x_mins_to_update', return_value=[]):
            result = bar_min.RcquantMinBarDownloader._backfill_sec_min(
                updater, force=True, first_n=-1, **kwargs,
            )
        return updater, result

    def test_default_and_explicit_limits(self):
        for kwargs, expected in [({}, 5), ({'max_days': 2}, 2), ({'max_days': '2'}, 2),
                                 ({'max_days': 0}, 0), ({'max_days': -1}, 0)]:
            with self.subTest(kwargs=kwargs):
                updater, _ = self.run_backfill(kwargs)
                self.assertEqual(updater.rcquant_bar_min.call_count, expected)
                self.assertEqual([call.args[0] for call in updater.rcquant_bar_min.call_args_list],
                                 list(range(7, 7 - expected, -1)))

    def test_unlimited_values(self):
        for value in [None, '', ' ', 'none', 'null', ' NoNe ', ' NULL ']:
            with self.subTest(value=value):
                updater, _ = self.run_backfill({'max_days': value})
                self.assertEqual(updater.rcquant_bar_min.call_count, 7)

    def test_unlimited_still_stops_at_deadline(self):
        updater, result = self.run_backfill({'max_days': None}, deadline=True)
        updater.rcquant_bar_min.assert_not_called()
        self.assertEqual(result, bar_min.Base.UpdateFlag.SKIPPED)


if __name__ == '__main__':
    unittest.main()
