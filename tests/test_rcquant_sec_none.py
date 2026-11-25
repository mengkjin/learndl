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

from src.data.download.other_source.rcquant import bar_min
from src.data.download.other_source.rcquant import sec_none_migration as migration
from src.data.download.other_source.rcquant.initializer import QuotaExceeded
from src.data.download.other_source.rcquant.migration_factors import (
    EMPTY_EVENTS, PRICE, Events, event_series, inverted_prices, restore_stock_day, round_half_up,
)


def minute_frame(date, *, price=10., volume=100.):
    day = pd.Timestamp(str(date))
    count = 240
    times = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq='min').append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=1), periods=120, freq='min')
    )
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
                self.assertIsNotNone(actual)
                price, shares, scaled = actual
                np.testing.assert_array_equal(price, raw_prices)
                np.testing.assert_array_equal(shares, [100., 100.])
                self.assertEqual(scaled, expected_scaled)

    def test_cent_half_up_and_ambiguous_legacy_inverse(self):
        np.testing.assert_array_equal(round_half_up(np.array([1.005, 2.675]), 100), [1.01, 2.68])
        self.assertIsNone(inverted_prices(np.array([0.01005]), 200.))
        np.testing.assert_array_equal(inverted_prices(np.array([8.]), 1.25), [10.])


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
        self.events = {method: {} for method in migration.METHODS}
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

    def test_split_vwap_failure_keeps_entire_legacy_date_and_records_evidence(self):
        native = minute_frame(self.date)
        native['order_book_id'] = '000001.XSHE'
        native.loc[0, 'amount'] = 1040.
        self.job.direct_none = Mock(return_value=native)
        with patch.object(migration.DB, 'load', return_value=pd.DataFrame()):
            self.job.migrate_date(self.date, self.events)
            self.job.migrate_date(self.date, self.events)
        self.job.direct_none.assert_called_once()
        self.assertIn('VWAP violations', self.job.read('blocked', self.date))
        audit = pd.read_feather(self.job.state / 'vwap_failures' / f'{self.date}.feather')
        self.assertEqual(audit.iloc[0]['vwap'], 10.4)
        pd.testing.assert_frame_equal(pd.read_feather(self.raw_path), self.legacy)

    def test_factor_path_publishes_without_a_minute_download(self):
        self.events['get_ex_factor']['000001.XSHE'] = Events(
            np.array([20250423]), np.array([1.25]),
        )
        daily = pd.DataFrame({'secid': [1], **{col: [10.] for col in PRICE}, 'volume': [24.]})
        with patch.object(migration.DB, 'load', return_value=daily), \
                patch.object(self.job, 'direct_none') as direct_none:
            self.job.migrate_date(self.date, self.events)
            self.job.migrate_date(self.date, self.events)
        direct_none.assert_not_called()
        self.assertEqual(self.job.read('done', self.date)['price_and_original_volume'], 1)
        minute = pd.read_feather(self.raw_path)
        self.assertTrue(minute['close'].eq(10.).all())
        self.assertTrue(minute['volume'].eq(100.).all())
        np.testing.assert_array_equal(minute['num_trades'], self.legacy['num_trades'])

    def test_interrupted_publish_keeps_pre_backup_and_does_not_adjust_twice(self):
        native = minute_frame(self.date)
        self.job.stage(self.date, native.assign(_adjust_type='none'), {'direct_none': 1})
        prepared = self.job.read('prepared', self.date)
        rename = migration.os.replace

        def interrupted(source, target):
            if Path(target).parent.name == 'min':
                raise OSError('interrupted after raw publish')
            rename(source, target)

        with patch.object(migration.os, 'replace', side_effect=interrupted), self.assertRaises(OSError):
            self.job.migrate_date(self.date, self.events)
        self.assertTrue(bar_min.is_none_min(pd.read_feather(self.raw_path)))
        original = self.raw_path.parent / '.pre_backup' / self.raw_path.name
        pd.testing.assert_frame_equal(pd.read_feather(original), self.legacy)
        self.job.migrate_date(self.date, self.events)
        self.assertEqual(self.job.read('done', self.date), {'direct_none': 1})
        self.assertEqual(self.job.read('prepared', self.date), prepared)
        totals = sum(native['num_trades'])
        for item in prepared['outputs'][1:]:
            self.assertEqual(pd.read_feather(item['target'])['num_trades'].sum(), totals)
        with patch.object(self.job, 'publish') as publish:
            self.job.migrate_date(self.date, self.events)
            publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
