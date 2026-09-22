"""Check alias storage and missing-date backfills without remote access."""
import importlib.util
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


spec = importlib.util.spec_from_file_location(
    'huayuan_backfill', Path(__file__).resolve().parents[1] / 'scripts/2_data/6_backfill_huayuan.py'
)
assert spec and spec.loader
backfill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill_module)


class HuayuanBackfillTest(unittest.TestCase):
    def setUp(self):
        self.downloader = MagicMock(
            start_date=20171229, end_date=99991231, factor_set='scores_v1',
            date_col='trade_dt', local_name='cmm', db_key='huayuan.cmm',
        )
        conn = self.downloader.connection.get_connection.return_value.__enter__.return_value
        conn.execute.return_value.scalars.return_value.all.return_value = [
            date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18),
        ]
        self.downloader.query_factor_values.return_value = pd.DataFrame({
            'date': [20260916, 20260917, 20260918], 'secid': [1, 1, 1], 'cmm': [0.1, 0.2, 0.3],
        })
        self.db = MagicMock()
        self.db.dates.return_value = [20260917]
        self.db.save.return_value = True

    def run_backfill(self, **kwargs):
        backfill_module.backfill(self.downloader, self.db, end=20260918, **kwargs)

    def test_missing_dates_do_not_overwrite_interior_existing_date(self):
        self.run_backfill()
        self.assertEqual([c.args[3] for c in self.db.save.call_args_list], [20260916, 20260918])
        for call in self.db.save.call_args_list:
            self.assertEqual(call.args[2], 'huayuan.cmm')
            self.assertFalse(call.kwargs['overwrite'])
        self.downloader.connection.close.assert_called_once()

    def test_overwrite_includes_existing_dates(self):
        self.run_backfill(overwrite=True)
        self.assertEqual(self.db.save.call_count, 3)
        self.assertTrue(all(c.kwargs['overwrite'] for c in self.db.save.call_args_list))

    def test_dry_run_does_not_write(self):
        self.run_backfill(dry_run=True)
        self.db.save.assert_not_called()

    def test_failed_or_incomplete_query_does_not_write(self):
        original = self.downloader.query_factor_values.return_value
        for data in [None, original.iloc[:1]]:
            with self.subTest(data=data):
                self.downloader.query_factor_values.return_value = data
                with self.assertRaises(RuntimeError):
                    self.run_backfill()
                self.db.save.assert_not_called()

    def test_failed_save_raises(self):
        self.db.save.return_value = False
        with self.assertRaises(RuntimeError):
            self.run_backfill()

    def test_configured_aliases(self):
        from src.data.download.sellside.from_sql import SellsideSQLDownloader
        expected = {'huayuan.cmm': 'scores_v1', 'huayuan.scores_v5': 'pred_alpha',
                    'huayuan.scores_v0': 'scores_v0'}
        for key, downloader in SellsideSQLDownloader.default_factors(list(expected)).items():
            self.assertEqual(downloader.factor_set, expected[key])
            self.assertEqual(downloader.db_key, key)
            raw = pd.DataFrame({'TRADE_DT': ['2026-09-18'], 'S_INFO_WINDCODE': ['000001.SZ'],
                                'FACTOR_VALUE': [0.5]})
            data = downloader.df_process(raw)
            self.assertIn(key.split('.')[1], data.columns)


if __name__ == '__main__':
    unittest.main()
