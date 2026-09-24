"""Sellside all-null days stay stored, and reads as-of the last usable day."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.proj import Dates
from src.data.download.sellside.from_sql import SellsideSQLDownloader
from src.data.download.sellside.valid_dates import (
    apply_validity,
    asof_map,
    finite_value_count,
    load_valid_dates,
    stamp_asof,
)


class SellsideValidDatesTest(unittest.TestCase):
    def test_all_null_day_is_saved_and_excluded_from_index(self):
        saved: list[int] = []

        def fake_save(df, src, key, date, **kwargs):
            del df, src, key, kwargs
            saved.append(int(date))
            return True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_index_path(db_key, db_src='sellside'):
                return root / db_src / db_key / 'valid_dates.feather'

            downloader = SellsideSQLDownloader(
                'huayuan', 'pred_alpha', 'trade_dt', 20171229, 99991231, '%Y-%m-%d',
                local_name='scores_v5',
            )
            frame = pd.DataFrame({
                'date': [20260105, 20260106, 20260106],
                'secid': [1, 1, 2],
                'scores_v5': [0.4, np.nan, np.nan],
            })
            with patch('src.data.download.sellside.from_sql.DB.save', fake_save), \
                    patch('src.data.download.sellside.valid_dates.index_path', fake_index_path):
                written = downloader.save_data(frame)
                indexed = load_valid_dates('huayuan.scores_v5')
            self.assertEqual(written, 2)
            self.assertEqual(saved, [20260105, 20260106])
            self.assertIsNotNone(indexed)
            assert indexed is not None
            self.assertEqual(indexed.tolist(), [20260105])
            # dates-mode missing set is file dates, so the empty day is not fetched again.
            pending = Dates([20260105, 20260106]).diff(saved)
            self.assertEqual(len(pending), 0)

    def test_overwriting_a_valid_day_with_nulls_drops_the_index_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_root = Path(tmp)

            def fake_index_path(db_key, db_src='sellside'):
                del db_key, db_src
                return path_root / 'valid_dates.feather'

            with patch('src.data.download.sellside.valid_dates.index_path', fake_index_path):
                apply_validity('huayuan.scores_v5', {20260105: True, 20260106: True})
                apply_validity('huayuan.scores_v5', {20260106: False})
                indexed = load_valid_dates('huayuan.scores_v5')
            self.assertIsNotNone(indexed)
            assert indexed is not None
            self.assertEqual(indexed.tolist(), [20260105])

    def test_asof_does_not_look_forward_and_shares_one_source(self):
        valid = np.array([20260105, 20260108], dtype=np.int64)
        mapping = asof_map(np.array([20260104, 20260106, 20260107, 20260108]), valid)
        self.assertNotIn(20260104, mapping)
        self.assertEqual(mapping[20260106], 20260105)
        self.assertEqual(mapping[20260107], 20260105)
        self.assertEqual(mapping[20260108], 20260108)
        loaded = pd.DataFrame({
            'secid': [1, 2],
            'date': [20260105, 20260105],
            'hy_scores_v5': [0.1, 0.2],
        })
        shared = {day: 20260105 for day in (20260105, 20260106, 20260107)}
        stamped = stamp_asof(loaded, shared)
        self.assertEqual(stamped['date'].nunique(), 3)
        self.assertEqual(len(stamped), 6)
        copied = stamped.groupby('date')['hy_scores_v5'].mean()
        self.assertTrue(np.allclose(copied.to_numpy(), 0.15))

    def test_finite_count_ignores_identifiers(self):
        frame = pd.DataFrame({'secid': [1, 2], 'scores_v5': [np.nan, np.nan]})
        self.assertEqual(finite_value_count(frame), 0)
        frame.loc[0, 'scores_v5'] = 1.0
        self.assertEqual(finite_value_count(frame), 1)


class SellsideFactorAsofTest(unittest.TestCase):
    def test_loads_reads_each_source_once_and_stamps_request_dates(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import hy_scores_v5

        calls: list[list[int]] = []

        def fake_valid(db_key, db_src='sellside'):
            del db_key, db_src
            return np.array([20260105], dtype=np.int64)

        def fake_loads(cls, src, key, dates, col=None, closest=False):
            del cls, src, key, col, closest
            calls.append([int(day) for day in Dates(dates).dates])
            return pd.DataFrame({
                'secid': [1, 2],
                'date': [20260105, 20260105],
                'hy_scores_v5': [0.1, 0.2],
            })

        with patch('src.data.download.sellside.valid_dates.load_valid_dates', fake_valid), \
                patch.object(hy_scores_v5, 'loads_from_db', classmethod(fake_loads)):
            out = hy_scores_v5.Loads([20260106, 20260107, 20260105])
        self.assertEqual(calls, [[20260105]])
        self.assertEqual(sorted(out['date'].unique().tolist()), [20260105, 20260106, 20260107])
        self.assertEqual(len(out), 6)


if __name__ == '__main__':
    unittest.main()
