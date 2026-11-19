"""Price publication must respect the complete predecessor chain."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import pandas as pd
from src.data.util.prices import adjusted as prices
from src.proj import Dates


class PriceChainTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        def path(source, key, date=None):
            return self.root / source / key / (str(date) + '.feather')
        self.path = path
        def load(source, key, date=None, **kwargs):
            p = path(source, key, date)
            return pd.read_feather(p) if p.exists() else pd.DataFrame()
        self.stack.enter_context(patch.object(prices.DB, 'path', side_effect=path))
        self.stack.enter_context(patch.object(prices.DB, 'load', side_effect=load))
        self.stack.enter_context(patch.object(prices.DB, 'dates', side_effect=lambda source,key: Dates([int(p.stem) for p in (self.root/source/key).glob('*.feather')])))
        self.stack.enter_context(patch.object(prices, 'first_date', return_value=20041230))
        self.stack.enter_context(patch.object(prices, 'listing_description', return_value=pd.DataFrame(columns=['secid','list_dt','delist_dt'])))
        self.stack.enter_context(patch.object(prices.Logger, 'stdout'))

    def raw(self, date, ids=(1,), values=(10.,)):
        frame = pd.DataFrame({'secid': ids, 'adjfactor': 1., **{c: values for c in prices.PRICE_COLS}})
        prices.atomic_frame(frame, self.path('trade_ts', 'day', date))
        return frame

    def test_first_update_backtracks_and_carries_halted_security(self):
        self.raw(20041230, (1,2), (10.,20.))
        self.raw(20041231)
        self.assertTrue(prices.ensure_adjprice(20041231))
        result = pd.read_feather(self.path('trade_ts','adjprice',20041231)).set_index('secid')
        self.assertEqual(result.loc[2,'close'],20.)
        self.assertFalse(prices.ensure_adjprice(20041231, reason='custom'))

    def test_missing_predecessor_never_becomes_a_new_boundary(self):
        self.raw(20041231)
        with self.assertRaisesRegex(ValueError, '20041230'):
            prices.ensure_adjprice(20041231)
        self.assertFalse(self.path('trade_ts','adjprice',20041231).exists())

    def test_calculation_rejects_empty_previous_outside_boundary(self):
        self.raw(20041231)
        with self.assertRaisesRegex(ValueError, 'previous trading day'):
            prices.calc_adjprice(20041231, previous=pd.DataFrame())

    def test_revision_propagates_forward_and_concurrent_requests_skip(self):
        self.raw(20041230, (1,2), (10.,20.))
        self.raw(20041231)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(prices.ensure_adjprice, [20041231,20041231]))
        self.assertEqual(sorted(results),[False,True])
        revised = self.raw(20041230, (1,2), (10.,30.))
        prices.publish_day(revised,20041230)
        result = pd.read_feather(self.path('trade_ts','adjprice',20041231)).set_index('secid')
        self.assertEqual(result.loc[2,'close'],30.)

    def test_invalid_source_and_failed_publish_preserve_output(self):
        self.raw(20041230)
        prices.ensure_adjprice(20041230)
        path = self.path('trade_ts','adjprice',20041230)
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            prices.publish_day(pd.DataFrame(),20041230)
        with patch.object(prices.os,'replace', side_effect=OSError('failed')):
            with self.assertRaises(OSError):
                prices.ensure_adjprice(20041230,force=True)
        self.assertEqual(path.read_bytes(),original)

    def test_gap_is_not_treated_as_a_holiday_or_new_seed(self):
        self.raw(20041230)
        prices.ensure_adjprice(20041230)
        self.raw(20050104)
        with self.assertRaisesRegex(ValueError, '20041231'):
            prices.ensure_adjprice(20050104)
        self.assertFalse(self.path('trade_ts','adjprice',20050104).exists())

    def test_custom_check_skips_verified_output(self):
        from src.data.update.custom.adjprice import AdjPriceUpdater
        from src.proj import Base
        self.raw(20041230)
        prices.ensure_adjprice(20041230)
        self.assertEqual(AdjPriceUpdater.proceed_update(20041230,20041230),Base.UpdateFlag.SKIPPED)

    def test_publication_validates_before_any_write(self):
        with patch.object(prices,'atomic_frame') as write:
            with self.assertRaises(ValueError):
                prices.publish_day(pd.DataFrame({'secid':[1]}),20041230)
            write.assert_not_called()

    def test_raw_publish_failure_cannot_trigger_derived_writes(self):
        frame = self.raw(20041230)
        path = self.path('trade_ts', 'day', 20041230)
        original = path.read_bytes()
        with patch.object(prices, 'atomic_frame', side_effect=OSError('raw save failed')), \
                patch.object(prices, '_ensure_adjprice') as derived:
            with self.assertRaisesRegex(OSError, 'raw save failed'):
                prices.publish_day(frame.assign(close=11.), 20041230)
            derived.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_raw_is_persistent_before_derived_builder_runs(self):
        frame = pd.DataFrame({'secid': [1], 'adjfactor': [1.], **{c: [10.] for c in prices.PRICE_COLS}})
        def check_raw(date, **kwargs):
            pd.testing.assert_frame_equal(pd.read_feather(self.path('trade_ts', 'day', date)), frame)
            return True
        with patch.object(prices, '_ensure_adjprice', side_effect=check_raw) as derived:
            prices.publish_day(frame, 20041230)
        derived.assert_called_once()

    def test_unchanged_raw_publication_reports_skip_but_repairs_missing_prices(self):
        frame = self.raw(20041230)
        self.assertFalse(prices.publish_day(frame, 20041230))
        self.assertTrue(self.path('trade_ts', 'adjprice', 20041230).exists())
        self.assertTrue(prices.publish_day(frame.assign(close=11.), 20041230))

    def test_bad_dates_and_empty_prices_never_publish_raw_data(self):
        valid = pd.DataFrame({'secid': [1], 'adjfactor': [1.], **{c: [10.] for c in prices.PRICE_COLS}})
        for date, frame in ((20050101, valid), (20041230, valid.assign(trade_date=20041231)),
                            (20041230, valid.assign(**{c: float('nan') for c in prices.PRICE_COLS}))):
            with self.subTest(date=date), patch.object(prices, 'atomic_frame') as write:
                with self.assertRaises(ValueError):
                    prices.publish_day(frame, date)
                write.assert_not_called()

    def test_listing_rejects_bad_revisions_without_losing_valid_reference(self):
        from src.data.util.prices.listing import update_listing_prices
        prices.atomic_frame(pd.DataFrame({'secid': [1], 'list_dt': [20041230]}),
                            self.path('information_ts', 'description'))
        frame = self.raw(20041230).assign(preclose=8.)
        prices.atomic_frame(frame, self.path('trade_ts', 'day', 20041230))
        self.assertTrue(update_listing_prices())
        output = self.path('information_ts', 'list_day_price')
        original = output.read_bytes()
        for invalid in (frame.drop(columns='preclose'), frame.assign(preclose=-1.), pd.concat([frame, frame])):
            prices.atomic_frame(invalid, self.path('trade_ts', 'day', 20041230))
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                update_listing_prices()
            self.assertEqual(output.read_bytes(), original)
        self.path('trade_ts', 'day', 20041230).unlink()
        with self.assertRaisesRegex(ValueError, 'source missing'):
            update_listing_prices()
        self.assertEqual(output.read_bytes(), original)

    def test_query_detects_revision_before_unchanged_immediate_predecessor(self):
        self.raw(20041230, (1, 2), (10., 20.))
        self.raw(20041231)
        self.raw(20050104)
        prices.ensure_adjprice(20050104)
        self.raw(20041230, (1, 2), (10., 35.))
        prices.ensure_adjprice(20050104)
        result = pd.read_feather(self.path('trade_ts', 'adjprice', 20050104)).set_index('secid')
        self.assertEqual(result.loc[2, 'close'], 35.)

    def test_query_detects_missing_ancestor_behind_valid_immediate_predecessor(self):
        self.raw(20041230, (1, 2), (10., 20.))
        self.raw(20041231)
        self.raw(20050104)
        prices.ensure_adjprice(20050104)
        self.path('trade_ts', 'adjprice', 20041230).unlink()
        self.assertTrue(prices.ensure_adjprice(20050104))
        self.assertTrue(self.path('trade_ts', 'adjprice', 20041230).exists())

    def test_frozen_reads_never_publish_snapshot_inputs_to_live_files(self):
        self.raw(20041230)
        prices.ensure_adjprice(20041230)
        output = self.path('trade_ts', 'adjprice', 20041230)
        original = output.read_bytes()
        with patch.object(prices, 'snapshot_active', return_value=True), patch.object(prices, 'atomic_frame') as write:
            self.assertFalse(prices.ensure_adjprice(20041230))
            with self.assertRaisesRegex(RuntimeError, 'read snapshot'):
                prices.ensure_adjprice(20041230, force=True)
            write.assert_not_called()
        self.assertEqual(output.read_bytes(), original)
        output.unlink()
        with patch.object(prices, 'snapshot_active', return_value=True):
            with self.assertRaisesRegex(ValueError, 'frozen adjprice'):
                prices.ensure_adjprice(20041230)
        self.assertFalse(output.exists())
