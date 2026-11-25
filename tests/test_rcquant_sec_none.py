"""Equity cutover, corporate-action inversion and migration recovery."""
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl

from src.data.download.other_source.rcquant import bar_min
from src.data.download.other_source.rcquant import sec_none_migration as migration
from src.data.download.other_source.rcquant.initializer import QuotaExceeded
from src.data.download.other_source.rcquant.migration_factors import (
    EMPTY_EVENTS, PRICE, Events, event_series, inverted_prices, restore_stock_day, round_half_up,
)
from src.data.download.other_source.rcquant.sec_history_polars import FactorHistory, _round, vwap_audit, with_vwap


def minute_frame(date, *, price=10., volume=100.):
    day = pd.Timestamp(str(date))
    count = 240
    times = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq='min').append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq='min')
    )
    times = pd.DatetimeIndex(times)
    return pd.DataFrame({
        'code': '000001.SZ', 'datetime': times, 'time': times.strftime('%H%M'),
        'date': str(date), 'open': price, 'high': price, 'low': price, 'close': price,
        'volume': volume, 'amount': 1000., 'num_trades': np.arange(count) % 7 + 1.,
    })


class FactorsTest(unittest.TestCase):
    def test_uses_ex_date_and_recomputes_split_cumulative(self):
        frame = pd.DataFrame({
            'order_book_id': ['000001.XSHE'] * 2,
            'ex_dividend_date': pd.to_datetime(['20200103', '20210118']),
            'book_closure_date': pd.to_datetime(['20200102', '20210115']),
            'split_coefficient_to': [12., 15.], 'split_coefficient_from': 10.,
            'cum_factor': [1.2, 1.5],
        })
        events = event_series('get_split', frame)['000001.XSHE']
        self.assertEqual(events.on(20200102), 1.)
        self.assertEqual(events.on(20200103), 1.2)
        self.assertAlmostEqual(events.on(20210118), 1.8)

    def test_cash_only_and_split_are_independent(self):
        prices = np.array([[8., 8., 8., 8.], [8., 8., 8., 8.]])
        raw_prices = np.full((2, 4), 10.)
        day = np.array([10., 10., 10., 10., 200.])
        event = Events(np.array([20200106]), np.array([1.25]))
        split = Events(np.array([20200106]), np.array([1.2]))
        for input_volume, splits, expected_scaled in [
            (np.array([100., 100.]), EMPTY_EVENTS, False),
            (np.array([100., 100.]), split, False),
            (np.array([120., 120.]), split, True),
        ]:
            with self.subTest(volume=input_volume, scaled=expected_scaled):
                actual = restore_stock_day(prices, input_volume, day, 20200103, event, splits)
                assert actual is not None
                price, shares, scaled = actual
                np.testing.assert_array_equal(price, raw_prices)
                np.testing.assert_array_equal(shares, [100., 100.])
                self.assertEqual(scaled, expected_scaled)

    def test_cent_half_up_and_ambiguous_legacy_inverse(self):
        np.testing.assert_array_equal(round_half_up(np.array([1.005, 2.675]), 100), [1.01, 2.68])
        actual = pl.DataFrame({'price': [1.005, 2.675]}).select(_round(pl.col('price'), 2)).to_series()
        self.assertEqual(actual.to_list(), [1.01, 2.68])
        self.assertIsNone(inverted_prices(np.array([0.01005]), 200.))
        np.testing.assert_array_equal(inverted_prices(np.array([8.]), 1.25), [10.])

    def test_vwap_audit_records_signed_gaps_and_excludes_zero_volume(self):
        frame = pl.DataFrame({'secid': [1, 1, 1], 'minute': [0, 1, 2],
                              **{name: [10.] * 3 for name in PRICE}, 'volume': [100., 100., 0.],
                              'amount': [1000.4, 996., 0.], 'num_trades': [1, 1, 0]})
        rows = vwap_audit(with_vwap(frame, 'open'), 1)
        self.assertEqual(rows['minute'].to_list(), [0, 1])
        np.testing.assert_allclose(rows['gap_yuan'].to_list(), [.004, -.04])


class CutoverTest(unittest.TestCase):
    def test_existing_pre_cache_is_replaced_and_all_periods_are_rebuilt(self):
        date = bar_min.SEC_NONE_START
        old = minute_frame(date, price=8.)
        fresh = minute_frame(date)
        provider = fresh.rename(columns={'code': 'order_book_id', 'amount': 'total_turnover'})
        provider['order_book_id'] = '000001.XSHE'
        provider = provider.drop(columns=['date', 'time']).set_index(['order_book_id', 'datetime'])
        updater = bar_min.RcquantMinBarDownloader()
        with tempfile.TemporaryDirectory() as folder, patch.object(bar_min, 'RQ_PATH', Path(folder)), \
                patch.object(bar_min.RQInitializer, 'init', return_value=True), \
                patch.object(bar_min, '_save_bar') as save, \
                patch.object(bar_min, 'rcquant_instrument_list',
                             return_value=pd.DataFrame({'code': ['000001.XSHE'], 'is_active': [True]})), \
                patch.object(bar_min.rqdatac, 'get_price', return_value=provider) as get_price:
            bar_min.write_min(old, date, migration.SEC)
            self.assertTrue(updater.rcquant_bar_min(date, migration.SEC))
            self.assertEqual(get_price.call_args.kwargs['adjust_type'], 'none')
            self.assertEqual([call.args[2] for call in save.call_args_list],
                             ['min', '5min', '10min', '15min', '30min', '60min'])
            for call in save.call_args_list:
                self.assertEqual(call.args[0]['num_trades'].sum(), fresh['num_trades'].sum())
            self.assertTrue(bar_min.is_none_min(bar_min.load_min(date, migration.SEC)))
            self.assertTrue(updater.rcquant_bar_min(date, migration.SEC))
            get_price.assert_called_once()


class MigrationTest(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(tmp)
        self.date = 20250422
        db = self.enterContext(closing(sqlite3.connect(self.path / 'progress.sqlite')))
        self.job = migration.Migration(self.path, self.path / 'exfactor', db, datetime(2026, 11, 25, 21))
        self.job.guard = Mock()
        self.events = FactorHistory.from_events(pl.DataFrame(), pl.DataFrame())
        self.db_path = self.enterContext(patch.object(migration, 'DBPath'))
        self.db_path.side_effect = lambda _, key: SimpleNamespace(
            path_exact=lambda date: self.path / key / f'{key}.{date}.feather',
            clear_date_cache=lambda: None,
        )
        self.raw_path = self.job.minute_path(self.date)
        self.raw_path.parent.mkdir()
        self.legacy = minute_frame(self.date, price=8.)
        self.legacy.to_feather(self.raw_path)
        with patch.object(bar_min, 'load_list', return_value=None):
            self.job.scan([self.date])

    def test_inventory_rechecks_calendar_after_an_empty_or_partial_scan(self):
        with self.assertRaisesRegex(ValueError, 'empty trading calendar'):
            self.job.scan([])
        later = self.date + 1
        manifest = self.job.scan([self.date, later])
        self.assertEqual(manifest['dates'], [self.date])
        self.assertEqual(manifest['missing'], [later])

    def test_inventory_reports_a_previously_scanned_archive_that_disappeared(self):
        source = self.job.require_record('scan', self.date)
        self.raw_path.unlink()
        manifest = self.job.scan([self.date])
        self.assertEqual(manifest['dates'], [])
        self.assertEqual(manifest['missing'], [self.date])
        self.job.direct_none = Mock()
        self.assertEqual(self.job.backfill_missing_raw(manifest['missing']), [])
        self.job.direct_none.assert_not_called()
        self.assertEqual(self.job.require_record('scan', self.date), source)

    def test_old_completion_does_not_skip_repair_and_missing_date_defers_datablocks(self):
        self.job.mark('complete', 'history', {'done': 1})
        with patch.object(migration, 'Dates', return_value=[self.date, self.date + 1]), \
                patch.object(self.job, 'backfill_missing_raw', return_value=[]), \
                patch.object(self.job, 'get_events', return_value=self.events) as fetch, \
                patch.object(self.job, 'migrate_date', return_value=True) as migrate, \
                patch.object(self.job, 'rebuild_datablocks') as rebuild:
            self.job.run()
        fetch.assert_called_once()
        migrate.assert_called_once_with(self.date, self.events, previously_written=False)
        rebuild.assert_not_called()

    def test_quota_batch_then_resume_and_skip_completed_batch(self):
        ids = ['000001.XSHE']
        self.job.remote = Mock(side_effect=[QuotaExceeded('quota exceeded'), None])
        with self.assertRaises(QuotaExceeded):
            self.job.factor_batch('get_split', ids, migration.START, migration.END)
        self.assertTrue(self.job.factor_batch('get_split', ids, migration.START, migration.END).empty)
        self.assertTrue(self.job.factor_batch('get_split', ids, migration.START, migration.END).empty)
        self.assertEqual(self.job.remote.call_count, 2)

    def test_daily_actions_resume_batches_and_record_empty_events(self):
        date = migration.BEGIN_NONE
        ids = [f'{i:06d}.XSHE' for i in range(1001)]
        universe = pd.DataFrame({'code': [*reversed(ids), ids[0]]})
        event = pd.DataFrame({
            'order_book_id': [ids[-1]], 'ex_date': [pd.Timestamp(str(date))],
            'ex_factor': [1.02], 'ex_cum_factor': [2.04],
        }).set_index('ex_date')
        self.job.remote = Mock(side_effect=[None, QuotaExceeded('quota exceeded'), event, None, None])
        with patch.object(bar_min, 'rcquant_instrument_list', return_value=universe) as get_universe:
            with self.assertRaises(QuotaExceeded):
                self.job.collect_daily_actions(date)
            self.assertTrue(self.job.collect_daily_actions(date))
            self.assertFalse(self.job.collect_daily_actions(date))
            get_universe.assert_called_once_with(date, migration.SEC, raise_on_quota=True)
        calls = self.job.remote.call_args_list
        self.assertEqual([len(call.args[1]) for call in calls], [1000, 1, 1, 1000, 1])
        for call in calls:
            self.assertEqual(call.kwargs, {'start_date': date, 'end_date': date, 'market': 'cn'})

        output = self.job.factor_dir / 'daily' / str(date)
        self.assertEqual(pd.read_feather(output / 'order_book_ids.feather')['order_book_id'].tolist(), ids)
        self.assertEqual(pd.read_feather(output / 'get_ex_factor.feather')['order_book_id'].tolist(), ids[-1:])
        self.assertTrue(pd.read_feather(output / 'get_split.feather').empty)
        self.assertEqual(self.job.read('daily_actions', date),
                         {'order_book_ids': 1001, 'get_ex_factor': 1, 'get_split': 0})

    def test_daily_actions_check_today_then_resume_missed_calendar_days(self):
        self.job.collect_daily_actions = Mock(side_effect=[True, False, False, True])
        self.assertTrue(self.job.collect_recent_actions(20261128))
        self.assertEqual([call.args[0] for call in self.job.collect_daily_actions.call_args_list],
                         [20261128, 20261125, 20261126, 20261127])

    def test_daily_actions_and_stock_minutes_share_instrument_cache(self):
        date = migration.BEGIN_NONE
        instrument = pd.DataFrame({'order_book_id': ['000001.XSHE'], 'status': ['Active']})
        self.job.remote = Mock(return_value=None)
        with patch.object(bar_min, 'RQ_PATH', self.path), \
                patch.object(bar_min.RQInitializer, 'init', return_value=True), \
                patch.object(bar_min.rqdatac, 'all_instruments', return_value=instrument) as get_instruments:
            self.job.collect_daily_actions(date)
            stored = bar_min.rcquant_instrument_list(date, migration.SEC)
        get_instruments.assert_called_once_with(type='CS', date=str(date))
        self.assertEqual(stored['code'].tolist(), ['000001.XSHE'])
        self.assertTrue(stored['is_active'].all())

    def test_native_vwap_outlier_is_audited_and_does_not_block_publication(self):
        native = minute_frame(self.date)
        native['order_book_id'] = '000001.XSHE'
        native.loc[0, 'amount'] = 1040.
        self.job.direct_none = Mock(return_value=native)
        self.job.mark('blocked', self.date, 'old VWAP gate')
        with patch.object(migration.DB, 'load_pl', return_value=pl.DataFrame()):
            self.assertTrue(self.job.migrate_date(self.date, self.events))
            self.assertTrue(self.job.migrate_date(self.date, self.events))
        self.job.direct_none.assert_called_once()
        done = self.job.require_record(self.job.repair_stage('done'), self.date)
        self.assertEqual(done['vwap_audit_1min'], 1)
        self.assertGreater(done['vwap_audit_xmin'], 0)
        audit = pl.read_ipc(self.job.state / 'vwap_audit' / f'{self.date}.feather', memory_map=False)
        self.assertEqual(audit.filter(pl.col('period') == 1)['vwap'].item(), 10.4)
        price_check = self.job.require_record(self.job.repair_stage('price_check'), self.date)
        self.assertEqual(price_check['previous_block'], 'old VWAP gate')
        self.assertEqual(price_check['input_ohlc'], [8.] * 4)
        self.assertTrue(bar_min.is_none_min(pd.read_feather(self.raw_path)))
        pd.testing.assert_frame_equal(pd.read_feather(self.job.backup_path(self.date)), self.legacy)

    def test_invalid_native_reply_does_not_poison_the_retry_cache(self):
        native = minute_frame(self.date).assign(order_book_id='000001.XSHE')
        invalid = native.copy()
        invalid.loc[0, 'low'] = 20.
        self.job.direct_none = Mock(side_effect=[invalid, native])
        with patch.object(migration.DB, 'load_pl', return_value=pl.DataFrame()):
            self.assertFalse(self.job.migrate_date(self.date, self.events))
            self.assertTrue(self.job.migrate_date(self.date, self.events))
        self.assertEqual(self.job.direct_none.call_count, 2)

    def test_missing_date_backfills_native_raw_and_builds_periods_without_rewriting_it(self):
        date = 20250423
        ids = [f'{i:06}.XSHE' for i in range(1, 6)] + [f'{i}.XSHG' for i in range(600001, 600006)]
        instruments = pd.DataFrame({'code': ids, 'is_active': True})
        provider = (pd.concat([minute_frame(date).assign(order_book_id=code) for code in ids],
                              ignore_index=True).drop(columns=['code', 'time', 'date']))
        self.job.direct_none = Mock(return_value=provider)
        migrate = self.job.migrate_date

        def publish_backfill_only(day, events, **kwargs):
            return day == date and migrate(day, events, **kwargs)

        with patch.object(migration, 'Dates', return_value=[self.date, date]), \
                patch.object(bar_min, 'rcquant_instrument_list', return_value=instruments), \
                patch.object(bar_min, 'load_list', return_value=None), \
                patch.object(self.job, 'get_events', return_value=self.events), \
                patch.object(migration.DB, 'load_pl', return_value=pl.DataFrame()), \
                patch.object(self.job, 'migrate_date', side_effect=publish_backfill_only), \
                patch.object(self.job, 'rebuild_datablocks') as rebuild:
            self.job.run()
            self.job.run()
        self.job.direct_none.assert_called_once_with(date, ids)
        self.assertTrue(bar_min.is_none_min(pd.read_feather(self.job.minute_path(date))))
        self.assertEqual(migration.fingerprint(self.job.minute_path(date)), self.job.read('raw_backfill', date))
        self.assertFalse(self.job.backup_path(date).exists())
        prepared = self.job.require_record(self.job.repair_stage('prepared'), date)
        self.assertTrue(prepared['preserve_raw'])
        self.assertEqual(len(prepared['outputs']), 6)
        for output in prepared['outputs']:
            self.assertTrue(Path(output['target']).is_file())
        self.assertTrue((self.job.factor_dir / f'{date}.feather').is_file())
        self.assertEqual(self.job.read('manifest', 'history')['missing'], [])
        rebuild.assert_not_called()

    def test_factor_path_publishes_without_a_minute_download(self):
        events = FactorHistory.from_events(pl.DataFrame({
            'order_book_id': ['000001.XSHE'], 'ex_date': [datetime(2025, 4, 23)],
            'ex_factor': [1.25], 'ex_cum_factor': [1.25],
        }), pl.DataFrame())
        daily = pl.DataFrame({'secid': [1], **{col: [10.] for col in PRICE}, 'volume': [24.]})
        with patch.object(migration.DB, 'load_pl', return_value=daily), \
                patch.object(self.job, 'direct_none') as direct_none:
            self.job.migrate_date(self.date, events)
            self.job.migrate_date(self.date, events)
        direct_none.assert_not_called()
        self.assertEqual(self.job.require_record(self.job.repair_stage('done'), self.date)
                         ['price_and_original_volume'], 1)
        minute = pd.read_feather(self.raw_path)
        self.assertTrue(minute['close'].eq(10.).all())
        self.assertTrue(minute['volume'].eq(100.).all())
        np.testing.assert_array_equal(minute['num_trades'], self.legacy['num_trades'])

    def test_mixed_adjustment_markers_prevent_a_second_adjustment(self):
        mixed = self.legacy.assign(_adjust_type='pre')
        mixed.loc[0, '_adjust_type'] = 'none'
        mixed.to_feather(self.raw_path)
        source = self.job.require_record('scan', self.date)
        source['raw'] = migration.fingerprint(self.raw_path)
        self.job.mark('scan', self.date, source)
        with patch.object(migration.DB, 'load_pl', return_value=pl.DataFrame()), \
                patch.object(self.job, 'direct_none') as download:
            self.assertFalse(self.job.migrate_date(self.date, self.events))
        download.assert_not_called()
        self.assertEqual(migration.fingerprint(self.raw_path), source['raw'])
        self.assertIn('mixes native', self.job.read(self.job.repair_stage('blocked'), self.date))

    def test_split_volume_roundtrip_requires_bar_precision_and_daily_total(self):
        from src.data.download.other_source.rcquant.sec_history_polars import daily_reference, restore_factors

        split = 1.3992772
        shares = 35300
        events = FactorHistory.from_events(pl.DataFrame(), pl.DataFrame({
            'order_book_id': ['000001.XSHE'],
            'ex_dividend_date': [datetime(2025, 4, 23)], 'book_closure_date': [datetime(2025, 4, 22)],
            'split_coefficient_from': [10.], 'split_coefficient_to': [10 * split],
        }))
        raw = pl.from_pandas(minute_frame(self.date, volume=round(shares * split, 4)))
        stocks = self.job.stock_mapping({'000001.SZ': '000001.XSHE'})
        raw = raw.join(stocks.select('code', 'order_book_id'), on='code', maintain_order='left')
        daily = pl.DataFrame({'secid': [1], **{col: [10.] for col in PRICE}, 'volume': [shares * 240 / 1000]})
        restored, scaled = restore_factors(raw, daily_reference(stocks, daily), events, self.date, stocks)
        self.assertEqual(scaled, 1)
        self.assertEqual(restored['volume'].unique().to_list(), [shares])
        bad_daily = daily.with_columns(pl.col('volume') + .001)
        bad_archive = raw.with_columns(pl.col('volume') + .0001)
        for bars, day in ((raw, bad_daily), (bad_archive, daily)):
            restored, _ = restore_factors(bars, daily_reference(stocks, day), events, self.date, stocks)
            self.assertTrue(restored.is_empty())

    def test_interrupted_publish_keeps_pre_backup_and_does_not_adjust_twice(self):
        native = minute_frame(self.date)
        self.job.stage(self.date, pl.from_pandas(native.assign(_adjust_type='none')), {'direct_none': 1})
        prepared = self.job.require_record(self.job.repair_stage('prepared'), self.date)
        rename = migration.os.replace

        def interrupted(source, target):
            rename(source, target)
            if Path(target) == self.raw_path:
                raise OSError('interrupted before recording raw publication')

        with patch.object(migration.os, 'replace', side_effect=interrupted), self.assertRaises(OSError):
            self.job.migrate_date(self.date, self.events)
        self.assertTrue(bar_min.is_none_min(pd.read_feather(self.raw_path)))
        self.assertIsNone(self.job.read(self.job.repair_stage('published'), f'{self.date}/{self.raw_path.name}'))
        original = self.raw_path.parent / '.pre_backup' / self.raw_path.name
        pd.testing.assert_frame_equal(pd.read_feather(original), self.legacy)
        self.assertEqual(self.job.repair_cohorts([self.date]), ([], [self.date]))
        self.assertEqual(self.job.original_path(self.date, previously_written=True), original)
        with patch.object(migration, 'Dates', return_value=[self.date]), \
                patch.object(self.job, 'get_events', return_value=self.events), \
                patch.object(self.job, 'prepare_raw') as restore, \
                patch.object(self.job, 'rebuild_datablocks') as rebuild:
            self.job.run()
        restore.assert_not_called()
        rebuild.assert_called_once_with([self.date])
        self.assertEqual(self.job.read(self.job.repair_stage('done'), self.date), prepared['counts'])
        totals = sum(native['num_trades'])
        for item in prepared['outputs'][1:]:
            self.assertEqual(pd.read_feather(item['target'])['num_trades'].sum(), totals)
        with patch.object(self.job, 'publish') as publish:
            self.job.migrate_date(self.date, self.events)
            publish.assert_not_called()

    def test_previous_version_is_repaired_again_from_verified_original(self):
        self.job.backup_path(self.date).parent.mkdir()
        self.raw_path.rename(self.job.backup_path(self.date))
        minute_frame(self.date).assign(_adjust_type='none').to_feather(self.raw_path)
        self.job.mark('done', self.date, {'direct_none': 1})
        self.job.mark('prepared', self.date, {'outputs': []})
        self.assertEqual(self.job.repair_cohorts([self.date]), ([], [self.date]))
        events = FactorHistory.from_events(pl.DataFrame({
            'order_book_id': ['000001.XSHE'], 'ex_date': [datetime(2025, 4, 23)],
            'ex_factor': [1.25], 'ex_cum_factor': [1.25],
        }), pl.DataFrame())
        daily = pl.DataFrame({'secid': [1], **{col: [10.] for col in PRICE}, 'volume': [24.]})
        with patch.object(migration.DB, 'load_pl', return_value=daily), \
                patch.object(self.job, 'direct_none') as native:
            self.assertTrue(self.job.migrate_date(self.date, events, previously_written=True))
        native.assert_not_called()
        price_check = self.job.require_record(self.job.repair_stage('price_check'), self.date)
        self.assertEqual(price_check['source'], 'backup')
        self.assertEqual(price_check['input_ohlc'], [8.] * 4)
        self.assertFalse(price_check['input_matches_tushare'])
        self.assertEqual(price_check['archive_ohlc'], [10.] * 4)
        self.assertTrue(price_check['archive_matches_tushare'])
        self.assertEqual(pd.read_feather(self.raw_path)['close'].unique().tolist(), [10.])
        pd.testing.assert_frame_equal(pd.read_feather(self.job.backup_path(self.date)), self.legacy)

    def test_datablock_rebuild_continues_from_completed_fit_and_predict_keys(self):
        def build(frame, key):
            path = self.path / frame / f'{key}.mmap'
            path.mkdir(parents=True)
            return {'path': str(path), 'dates': 1, 'last': self.date}

        with patch('src.proj.util.script.script_lock.ScriptLockMultiple'), \
                patch.object(self.job, 'rebuild_minute_block', side_effect=build) as rebuild:
            self.job.rebuild_datablocks([self.date])
            self.job.rebuild_datablocks([self.date])
        self.assertEqual([call.args for call in rebuild.call_args_list], [
            (frame, key) for frame in ('fit', 'predict') for key in migration.MINUTE_DATABLOCKS
        ])
        self.assertEqual(self.job.require_record(self.job.repair_stage('complete'), 'history')['dates'], 1)

    def test_datablock_rebuild_publishes_mmap_and_preserves_previous_values(self):
        from src.data.preprocess.processors import PrePros
        from src.data.util import DataBlock
        from src.proj.db.io.mmap import ArrayMemoryMap

        canonical = self.path / 'fit' / 'X_15m.mmap'
        index = {'secid': np.array([1]), 'date': np.array([self.date]), 'feature': np.array(['close'])}
        original = np.full((1, 1, 1, 1), 8., dtype=np.float32)
        migration.Save.mmap(original, canonical, index=index)
        new = SimpleNamespace(**index, empty=False, last_meaningful_date=self.date, values=original + 2)
        processor = Mock()
        processor.load_with_extension.return_value = new
        with patch.object(PrePros, 'get_processor', return_value=processor), \
                patch.object(DataBlock, 'path_preprocess', return_value=canonical):
            self.assertEqual(self.job.rebuild_minute_block('fit', '15m')['dates'], 1)
        processor.load_with_extension.assert_called_once_with(reconstruct=True)
        processor.save_norm.assert_called_once_with(new)
        np.testing.assert_array_equal(ArrayMemoryMap.load(canonical)['values'], new.values)
        backup = canonical.parent / f'.{migration.REPAIR}_backup' / canonical.name
        np.testing.assert_array_equal(ArrayMemoryMap.load(backup)['values'], original)


if __name__ == '__main__':
    unittest.main()
