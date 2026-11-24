"""Sellside all-null days stay stored, and reads as-of the last usable day."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.proj import DB, PATH, Dates
from src.data.download.sellside.from_sql import SellsideSQLDownloader
from src.data.quality.valid_dates import (
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
                    patch('src.data.quality.valid_dates.valid_values_path', fake_values_path):
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

            with patch('src.data.quality.valid_dates.valid_values_path', fake_values_path):
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

            with patch('src.data.quality.valid_dates.valid_values_path', fake_values_path), \
                    patch('src.data.quality.valid_dates.DB.dates', return_value=_Stored()), \
                    patch('src.data.quality.valid_dates.DB.load', fake_load):
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
    def test_primary_and_backup_use_latest_date_then_source_order(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import df_scores_v3

        primary, backup = 'dongfang.scores_v3', 'huayuan.scores_v3'
        valid = {
            primary: [20251231, 20260107],
            backup: [20260105, 20260107, 20260109],
        }
        calls = []

        def fake_loads(cls, src, key, dates, col=None, closest=False):
            calls.append((key, list(dates)))
            return pd.DataFrame({
                'secid': 1, 'date': dates,
                cls.factor_name: [day if key == primary else -day for day in dates],
            })

        requested = [20251230, 20251231, 20260105, 20260107, 20260108, 20260109]
        expected = {
            20251231: 20251231, 20260105: -20260105, 20260107: 20260107,
            20260108: 20260107, 20260109: -20260109,
        }
        with patch('src.data.quality.valid_dates.load_valid_dates',
                   side_effect=lambda key, *args, **kwargs: np.array(valid[key])), \
                patch.object(df_scores_v3, 'loads_from_db', classmethod(fake_loads)):
            frame = df_scores_v3.Loads(requested)
            self.assertEqual(frame.set_index('date')['df_scores_v3'].to_dict(), expected)
            self.assertCountEqual(calls, [
                (primary, [20251231, 20260107]), (backup, [20260105, 20260109]),
            ])
            factor = df_scores_v3()
            self.assertTrue(factor.load_factor(20251230).empty)
            for date, value in expected.items():
                with self.subTest(date=date):
                    single = factor.load_factor(date)
                    self.assertEqual(single.columns.tolist(), ['secid', 'df_scores_v3'])
                    self.assertEqual(single['df_scores_v3'].tolist(), [value])

    def test_stale_index_falls_back_to_another_source_or_an_older_date(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import df_scores_v3

        primary, backup = 'dongfang.scores_v3', 'huayuan.scores_v3'
        valid = {
            primary: [20260105, 20260107, 20260109],
            backup: [20260107, 20260112, 20260113],
        }
        stored = {
            primary: {20260105: 1.0, 20260107: np.nan},
            backup: {20260107: 2.0, 20260112: 3.0, 20260113: np.inf},
        }

        def fake_loads(cls, src, key, dates, col=None, closest=False):
            return pd.DataFrame(
                [(1, day, stored[key][day]) for day in dates if day in stored[key]],
                columns=['secid', 'date', cls.factor_name],
            )

        with patch('src.data.quality.valid_dates.load_valid_dates',
                   side_effect=lambda key, *args, **kwargs: np.array(valid[key])), \
                patch.object(df_scores_v3, 'loads_from_db', classmethod(fake_loads)):
            frame = df_scores_v3.Loads([20260106, 20260107, 20260109, 20260113])
        self.assertEqual(frame.set_index('date')['df_scores_v3'].to_dict(), {
            20260106: 1.0, 20260107: 2.0, 20260109: 2.0, 20260113: 3.0,
        })

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

        with patch('src.data.quality.valid_dates.load_valid_dates', fake_valid), \
                patch.object(hy_scores_v5, 'loads_from_db', classmethod(fake_loads)):
            out = hy_scores_v5.Loads([20260106, 20260107, 20260105])
        self.assertEqual(calls, [[20260105]])
        self.assertEqual(sorted(out['date'].unique().tolist()), [20260105, 20260106, 20260107])
        self.assertEqual(len(out), 6)

    def test_missing_valid_values_requires_explicit_backfill(self):
        from src.res.factor.defs.affiliate.level0.external.sellside import hy_scores_v5
        from src.data.quality.valid_dates import load_valid_dates, write_valid_values

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

            with patch('src.data.quality.valid_dates.valid_values_path', fake_values_path), \
                    patch('src.data.quality.valid_dates.DB.dates', return_value=_Stored()), \
                    patch('src.data.quality.valid_dates.DB.load', fake_scan), \
                    patch.object(hy_scores_v5, 'loads_from_db', classmethod(fake_loads)):
                with self.assertRaises(FileNotFoundError):
                    load_valid_dates('huayuan.scores_v5', required=True)
                self.assertEqual(scanned, [])
                backfill_valid_values('huayuan.scores_v5')
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


class SellsidePortfolioFreshnessTest(unittest.TestCase):
    def setUp(self):
        from src.res.factor.util.classes.alpha_model import AlphaComponent

        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(PATH, 'db', Path(directory)))
        self.enterContext(patch.object(AlphaComponent, '_cache_normalized', {}))
        self.enterContext(patch.object(AlphaComponent, '_cache_unnormalized', {}))

    def test_factor_alias_masks_expired_dates_even_after_a_fresh_read(self):
        from src.res.factor.calculator import StockFactorHierarchy
        from src.res.factor.defs.affiliate.level0.external.sellside import df_scores_v3
        from src.res.factor.util.classes.alpha_model import AlphaComposite
        from src.res.trading.trading_port import BacktestPort, TrackingPort

        primary, backup = 'dongfang.scores_v3', 'huayuan.scores_v3'
        DB.save(pd.DataFrame({'secid': [1, 2], 'pred': [1., 2.]}),
                'sellside', primary, 20240229, vb_level='never')
        DB.save(pd.DataFrame({'secid': [1, 2], 'scores_v3': [4., 3.]}),
                'sellside', backup, 20240402, vb_level='never')
        backfill_valid_values(primary)
        backfill_valid_values(backup)

        portfolio_alpha = AlphaComposite(['factor@df_scores_v3'])
        with patch.object(StockFactorHierarchy, 'get_factor', return_value=df_scores_v3):
            boundary = portfolio_alpha.get(20240329).item()
            batch = portfolio_alpha.get([20240329, 20240401, 20240402])
            repeat = portfolio_alpha.get(20240401).item()
            tracking = TrackingPort('freshness_probe', 'factor@df_scores_v3')
            with patch.object(tracking.Universe, 'get') as universe:
                self.assertTrue(tracking.build_portfolio(20240401, export=False).empty)
                universe.assert_not_called()
            backtest = BacktestPort('freshness_probe', 'factor@df_scores_v3',
                                    backtest=True, test_start=20240329)
            with patch.object(backtest, 'end_date', return_value=-1), \
                    patch.object(backtest, 'get_last_port'), \
                    patch.object(backtest.Universe, 'get'), \
                    patch('src.res.trading.trading_port.PortfolioBuilder') as builder:
                backtest.build_backward(20240402, export=False)
                builds = builder.return_value.setup.return_value.build.call_args_list
                self.assertEqual([call.args[0] for call in builds], [20240329, 20240402])

            self.assertEqual(df_scores_v3().load_factor(20240401)['df_scores_v3'].tolist(), [1., 2.])
            DB.save(pd.DataFrame({'secid': [1, 2], 'scores_v3': [6., 5.]}),
                    'sellside', backup, 20240401, vb_level='never')
            backfill_valid_values(backup)
            late = portfolio_alpha.get(20240401).item()

        self.assertTrue(np.isfinite(boundary.alpha_of([1, 2])).all())
        self.assertTrue(np.isnan(batch.get(20240401).alpha_of([1, 2])).all())
        self.assertTrue(np.isnan(repeat.alpha_of([1, 2])).all())
        self.assertGreater(batch.get(20240402).alpha_of([1, 2])[0],
                           batch.get(20240402).alpha_of([1, 2])[1])
        self.assertTrue(np.isfinite(late.alpha_of([1, 2])).all())

    def test_direct_alias_ignores_stale_or_all_null_columns_in_composites_and_screeners(self):
        from src.res.factor.fmp.generator.basic import BasicCreatorConfig
        from src.res.factor.util.classes.alpha_model import AlphaComponent, AlphaComposite, AlphaScreener, Amodel

        key = 'huatai.master_combined'
        direct = f'sellside@{key}@master_combined'
        other_column = f'sellside@{key}@secondary'
        DB.save(pd.DataFrame({'secid': [1, 2], 'master_combined': [1., 2.], 'secondary': [1., 2.]}),
                'sellside', key, 20240229, vb_level='never')
        DB.save(pd.DataFrame({'secid': [1, 2], 'master_combined': [np.nan, np.nan], 'secondary': [3., np.nan]}),
                'sellside', key, 20240329, vb_level='never')

        data = AlphaComponent(direct, normalize=False).get_alpha_model([20240329, 20240401])
        alternative = AlphaComponent(other_column, normalize=False).get_alpha_model(20240401)
        np.testing.assert_array_equal(data.get(20240329).alpha_of([1, 2]), [1., 2.])
        self.assertTrue(np.isnan(data.get(20240401).alpha_of([1, 2])).all())
        np.testing.assert_allclose(alternative.item().alpha_of([1, 2]), [3., np.nan], equal_nan=True)

        current = Amodel(20240401, np.array([1., 2.]), np.array([1, 4]), 'current')
        combined = AlphaComposite([direct, current]).get(20240401).item()
        np.testing.assert_array_equal(combined.secid, current.secid)
        np.testing.assert_allclose(combined.alpha, current.zscore().alpha)
        self.assertEqual(AlphaScreener([direct]).screened_pool(20240401, [1, 2]).size, 0)
        config = BasicCreatorConfig(sorter=direct, screener='self')
        self.assertIs(config.get_sorting_alpha(20240401, current), current)


if __name__ == '__main__':
    unittest.main()
