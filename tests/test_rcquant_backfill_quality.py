"""Historical equity backfill must stop at insufficient exchange coverage."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.data.download.other_source.rcquant import bar_min


def instruments():
    return pd.DataFrame({
        'code': [f'{600000 + i}.XSHG' for i in range(5)]
                + [f'{i + 1:06}.XSHE' for i in range(5)],
        'is_active': True,
    })


def minute_data():
    return instruments()[['code']].assign(close=10., datetime=pd.Timestamp('2009-01-05 09:31'))


class CoverageTest(unittest.TestCase):
    def test_five_distinct_stocks_per_exchange_pass_with_both_suffix_formats(self):
        for local in [False, True]:
            frame = minute_data()
            if local:
                frame['code'] = frame['code'].str.replace('.XSHG', '.SH', regex=False).str.replace('.XSHE', '.SZ', regex=False)
            bar_min.check_sec_backfill_quality(frame, instruments(), 20090105)

    def test_each_market_must_pass_and_duplicate_bars_do_not_count(self):
        for missing in [0, 9]:
            frame = minute_data().drop(index=missing)
            frame = pd.concat([frame] * 240, ignore_index=True)
            with self.assertRaisesRegex(bar_min.SecBackfillQualityError, 'SH=4, SZ=5|SH=5, SZ=4'):
                bar_min.check_sec_backfill_quality(frame, instruments(), 20090105)

    def test_unlisted_instruments_cannot_replace_missing_stock(self):
        frame = minute_data()
        frame.loc[0, 'code'] = '000300.XSHG'
        with self.assertRaisesRegex(bar_min.SecBackfillQualityError, 'SH=4, SZ=5'):
            bar_min.check_sec_backfill_quality(frame, instruments(), 20090105)

    def test_empty_and_invalid_prices_fail(self):
        for frame in [None, pd.DataFrame(), minute_data().drop(columns='close')]:
            with self.assertRaisesRegex(bar_min.SecBackfillQualityError, 'SH=0, SZ=0'):
                bar_min.check_sec_backfill_quality(frame, instruments(), 20090105)
        for value in [np.nan, np.inf, -np.inf, 0, -1, 'invalid']:
            frame = minute_data().astype({'close': object})
            frame.loc[0, 'close'] = value
            with self.subTest(value=value), self.assertRaisesRegex(bar_min.SecBackfillQualityError, 'SH=4, SZ=5'):
                bar_min.check_sec_backfill_quality(frame, instruments(), 20090105)


class DownloadQualityTest(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(bar_min, 'RQ_PATH', Path(folder)))
        self.enterContext(patch.object(bar_min.RQInitializer, 'init', return_value=True))
        self.enterContext(patch.object(bar_min, 'rcquant_instrument_list', return_value=instruments()))
        self.write = self.enterContext(patch.object(bar_min, 'write_min'))
        self.save = self.enterContext(patch.object(bar_min, '_save_bar'))
        self.updater = SimpleNamespace(indent=0, vb_level=0, rcquant_min_to_normal_min=lambda df, kind: df)

    def download(self, **kwargs):
        return bar_min.RcquantMinBarDownloader.rcquant_bar_min(
            self.updater, 20090105, bar_min.MinDataType.SEC, **kwargs,
        )

    def test_bad_download_and_empty_payload_are_not_saved(self):
        for frame in [None, pd.DataFrame(), minute_data().iloc[:5].rename(columns={'code': 'order_book_id'})]:
            with patch.object(bar_min, 'load_min', return_value=None), \
                 patch.object(bar_min, 'rq_get_price', return_value=frame), \
                 self.assertRaises(bar_min.SecBackfillQualityError):
                self.download(check_sec_quality=True)
        self.write.assert_not_called()
        self.save.assert_not_called()

    def test_bad_raw_cache_cannot_bypass_guard(self):
        with patch.object(bar_min, 'load_min', return_value=minute_data().iloc[:5]), \
             patch.object(bar_min, 'rq_get_price') as provider, \
             self.assertRaises(bar_min.SecBackfillQualityError):
            self.download(check_sec_quality=True)
        provider.assert_not_called()
        self.write.assert_not_called()
        self.save.assert_not_called()

    def test_valid_new_and_cached_data_are_saved(self):
        for cached in [False, True]:
            with patch.object(bar_min, 'load_min', return_value=minute_data() if cached else None), \
                 patch.object(bar_min, 'rq_get_price', return_value=minute_data().rename(columns={'code': 'order_book_id'})):
                self.assertTrue(self.download(check_sec_quality=True))
        self.assertEqual(self.write.call_count, 1)
        self.assertEqual(self.save.call_count, 2)

    def test_daily_download_preserves_existing_behavior(self):
        with patch.object(bar_min, 'load_min', return_value=minute_data().iloc[:1]):
            self.assertTrue(self.download())
        self.save.assert_called_once()


class BackfillStopTest(unittest.TestCase):
    def test_quality_failure_stops_before_older_dates_even_with_force(self):
        for good_days in [0, 1]:
            updater = MagicMock()
            updater.rcquant_bar_min.side_effect = [True] * good_days + [bar_min.SecBackfillQualityError('SH=5, SZ=4')]
            with patch.object(bar_min, 'MACHINE', updatable=True, belong_to_hfm=False), \
                 patch.object(bar_min, 'backfill_sec_dates', return_value=pd.Index([20090105, 20090106, 20090107])), \
                 patch.object(bar_min, '_backfill_deadline_hit', return_value=False), \
                 patch.object(bar_min, 'x_mins_to_update', return_value=[]) as aggregate:
                result = bar_min.RcquantMinBarDownloader._backfill_sec_min(
                    updater, force=True, first_n=-1, max_days=None,
                )
            calls = updater.rcquant_bar_min.call_args_list
            self.assertEqual([call.args[0] for call in calls], [20090107, 20090106][:good_days + 1])
            self.assertTrue(all(call.kwargs['check_sec_quality'] for call in calls))
            self.assertEqual(aggregate.call_count, good_days)
            self.assertEqual(result, bar_min.Base.UpdateFlag.SUCCESS if good_days else bar_min.Base.UpdateFlag.SKIPPED)
            self.assertIn('SH=5, SZ=4', updater.logger.conclude.call_args.args[0])

    def test_backfill_candidates_extend_to_2009(self):
        with patch.object(bar_min, 'src_start_date', return_value=20241101), \
             patch.object(bar_min.CALENDAR, 'td', return_value=20241031), \
             patch.object(bar_min, 'stored_dates', return_value=[]), \
             patch.object(bar_min, 'Dates') as dates:
            bar_min.backfill_sec_dates()
        dates.assert_called_once_with(20090101, 20241031)

    def test_quota_still_stops_backfill(self):
        updater = MagicMock()
        updater.rcquant_bar_min.side_effect = bar_min.QuotaExceeded('quota')
        with patch.object(bar_min, 'MACHINE', updatable=True, belong_to_hfm=False), \
             patch.object(bar_min, 'backfill_sec_dates', return_value=pd.Index([20090105, 20090106])), \
             patch.object(bar_min, '_backfill_deadline_hit', return_value=False):
            result = bar_min.RcquantMinBarDownloader._backfill_sec_min(
                updater, force=True, first_n=-1, max_days=None,
            )
        updater.rcquant_bar_min.assert_called_once()
        self.assertEqual(result, bar_min.Base.UpdateFlag.SKIPPED)


if __name__ == '__main__':
    unittest.main()
