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
    asof_map,
    backfill_valid_values,
    finite_value_count,
    load_valid_dates,
    read_valid_values,
    stamp_asof,
    upsert_valid_values,
    write_valid_values,
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

            stats_path = root / 'sellside' / '.data_stats' / 'valid_values' / 'huayuan.scores_v5.feather'

            def fake_values_path(db_key, db_src='sellside'):
                del db_key, db_src
                return stats_path

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
                    patch('src.data.download.sellside.valid_dates.valid_values_path', fake_values_path):
                written = downloader.save_data(frame)
                indexed = load_valid_dates('huayuan.scores_v5')
                table = read_valid_values(stats_path)
            self.assertEqual(written, 2)
            self.assertEqual(saved, [20260105, 20260106])
            self.assertIsNotNone(indexed)
            assert indexed is not None
            self.assertEqual(indexed.tolist(), [20260105])
            self.assertIsNotNone(table)
            assert table is not None
            self.assertEqual(table['date'].tolist(), [20260105, 20260106])
            self.assertEqual(table['secid_count'].tolist(), [1, 2])
            self.assertEqual(table['nan_count'].tolist(), [0, 2])
            self.assertEqual(table['is_valid'].tolist(), [True, False])
            # dates-mode missing set is file dates, so the empty day is not fetched again.
            pending = Dates([20260105, 20260106]).diff(saved)
            self.assertEqual(len(pending), 0)

    def test_overwriting_a_valid_day_with_nulls_drops_the_index_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path_root = Path(tmp)
            stats_path = path_root / 'valid_values.feather'

            def fake_values_path(db_key, db_src='sellside'):
                del db_key, db_src
                return stats_path

            def row(date, secid_count, nan_count, is_valid):
                return pd.DataFrame([{
                    'date': date,
                    'secid_count': secid_count,
                    'nan_count': nan_count,
                    'is_valid': is_valid,
                }])

            with patch('src.data.download.sellside.valid_dates.valid_values_path', fake_values_path):
                upsert_valid_values('huayuan.scores_v5', pd.concat([
                    row(20260105, 1, 0, True), row(20260106, 2, 0, True),
                ], ignore_index=True))
                upsert_valid_values('huayuan.scores_v5', row(20260106, 2, 2, False))
                indexed = load_valid_dates('huayuan.scores_v5')
                table = read_valid_values(stats_path)
            self.assertIsNotNone(indexed)
            assert indexed is not None
            self.assertEqual(indexed.tolist(), [20260105])
            self.assertIsNotNone(table)
            assert table is not None
            self.assertEqual(table['date'].tolist(), [20260105, 20260106])
            self.assertEqual(bool(table.loc[table['date'] == 20260106, 'is_valid'].iloc[0]), False)

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

    def test_missing_file_scans_history_and_a_gap_scans_only_the_gap(self):
        frames = {
            20260105: pd.DataFrame({'secid': [1], 'scores_v5': [0.4]}),
            20260106: pd.DataFrame({'secid': [1, 2], 'scores_v5': [np.nan, np.nan]}),
        }
        loaded: list[int] = []

        class _Stored:
            dates = np.array([20260105, 20260106], dtype=np.int64)

            def __len__(self):
                return 2

        def fake_load(src, key, date, **kwargs):
            del src, key, kwargs
            loaded.append(int(date))
            return frames[int(date)]

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / 'valid_values.feather'

            def fake_values_path(db_key, db_src='sellside'):
                del db_key, db_src
                return stats_path

            with patch('src.data.download.sellside.valid_dates.valid_values_path', fake_values_path), \
                    patch('src.data.download.sellside.valid_dates.DB.dates', return_value=_Stored()), \
                    patch('src.data.download.sellside.valid_dates.DB.load', fake_load):
                self.assertEqual(backfill_valid_values('huayuan.scores_v5'), 2)
                self.assertEqual(loaded, [20260105, 20260106])
                loaded.clear()
                self.assertEqual(backfill_valid_values('huayuan.scores_v5'), 0)
                self.assertEqual(loaded, [])
                table = read_valid_values(stats_path)
                assert table is not None
                kept = table.loc[table['date'] == 20260105]
                write_valid_values(stats_path, kept)
                self.assertEqual(backfill_valid_values('huayuan.scores_v5'), 1)
                self.assertEqual(loaded, [20260106])


class SellsideFactorAsofTest(unittest.TestCase):
    def test_loads_reads_each_source_once_and_stamps_request_dates(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import hy_scores_v5

        calls: list[list[int]] = []

        def fake_valid(db_key, db_src='sellside', required=False):
            del db_key, db_src, required
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

    def test_missing_valid_values_is_built_then_used_for_asof(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import hy_scores_v5
        from src.data.download.sellside.valid_dates import load_valid_dates, write_valid_values

        frames = {
            20260105: pd.DataFrame({'secid': [1], 'scores_v5': [0.4]}),
            20260106: pd.DataFrame({'secid': [1], 'scores_v5': [np.nan]}),
        }
        scanned: list[int] = []
        loaded_sources: list[list[int]] = []

        class _Stored:
            dates = np.array([20260105, 20260106], dtype=np.int64)

            def __len__(self):
                return 2

        def fake_scan(src, key, date, **kwargs):
            del src, key, kwargs
            scanned.append(int(date))
            return frames[int(date)]

        def fake_loads(cls, src, key, dates, col=None, closest=False):
            del cls, src, key, col, closest
            loaded_sources.append([int(day) for day in Dates(dates).dates])
            return pd.DataFrame({
                'secid': [1],
                'date': [20260105],
                'hy_scores_v5': [0.4],
            })

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / 'huayuan.scores_v5.feather'

            def fake_values_path(db_key, db_src='sellside'):
                del db_key, db_src
                return stats_path

            with patch('src.data.download.sellside.valid_dates.valid_values_path', fake_values_path), \
                    patch('src.data.download.sellside.valid_dates.DB.dates', return_value=_Stored()), \
                    patch('src.data.download.sellside.valid_dates.DB.load', fake_scan), \
                    patch.object(hy_scores_v5, 'loads_from_db', classmethod(fake_loads)):
                out = hy_scores_v5.Loads([20260106, 20260107])
                self.assertEqual(scanned, [20260105, 20260106])
                valid = load_valid_dates('huayuan.scores_v5', required=True)
                self.assertEqual(scanned, [20260105, 20260106])
                scanned.clear()
                write_valid_values(stats_path, pd.DataFrame([{
                    'date': 20260105, 'secid_count': 1, 'nan_count': 0, 'is_valid': True,
                }]))
                again = load_valid_dates('huayuan.scores_v5', required=True)
        self.assertEqual(scanned, [])
        self.assertEqual(loaded_sources, [[20260105]])
        self.assertEqual(out['date'].tolist(), [20260106, 20260107])
        self.assertIsNotNone(valid)
        assert valid is not None
        self.assertEqual(valid.tolist(), [20260105])
        self.assertIsNotNone(again)
        assert again is not None
        self.assertEqual(again.tolist(), [20260105])


if __name__ == '__main__':
    unittest.main()
