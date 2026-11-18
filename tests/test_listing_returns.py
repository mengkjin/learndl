"""Listing reference prices supplement close returns without changing VWAP/open."""
from __future__ import annotations

import threading
from contextlib import ExitStack
import unittest
from unittest.mock import patch
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader.data_vendor import DataVendor, DATAVENDOR
from src.data.loader.trade_data import TRADE
from src.data.util import DataBlock

__all__ = ['ListingReturnsTest']


class ListingReturnsTest(unittest.TestCase):
    def setUp(self):
        self.vendor: Any = object.__new__(DataVendor)
        self.vendor._list_price_lock = threading.RLock()
        self.vendor._lock = threading.Lock()
        self.vendor._list_day_price = None
        self.vendor.blocks_cache = {}
        self.vendor.__dict__['all_stocks'] = pd.DataFrame({
            'secid': [1, 2, 3], 'list_dt': [20260106, 20260106, 20260108]})
        self.day = pd.DataFrame({'secid': [1, 2], 'preclose': [10., 20.], 'adjfactor': [2., 1.]})

    def test_lazy_price_table_actual_factor_and_retry_missing(self):
        with patch('src.data.loader.data_vendor.DB.load', side_effect=[pd.DataFrame(), self.day]) as load:
            self.assertTrue(self.vendor.list_day_price.list_price.isna().all())
            load.assert_not_called()
            self.assertTrue(self.vendor.get_list_day_price(20260106, 20260106).adj_list_price.isna().all())
            table = self.vendor.get_list_day_price(20260106, 20260106)
            self.assertEqual(table.loc[1, 'adj_list_price'], 20.)
            self.assertEqual(table.loc[1, 'list_price'], 10.)
            self.vendor.get_list_day_price(20260106, 20260106)
            self.assertEqual(load.call_count, 2)
            self.assertTrue(np.isnan(self.vendor.list_day_price.loc[3, 'list_price']))
            table.loc[1, 'list_price'] = 999
            self.assertEqual(self.vendor.list_day_price.loc[1, 'list_price'], 10.)

    def test_clear_all_invalidates_listing_reference_cache(self):
        with patch('src.data.loader.data_vendor.DB.load', return_value=self.day):
            self.vendor.get_list_day_price(20260106, 20260106)
        self.vendor.day_quotes = {}
        self.vendor.day_secids = {}
        self.vendor.CUSTOM_DATA = {}
        with ExitStack() as stack:
            for name in ('INFO', 'TRADE', 'RISK', 'INDI', 'BS', 'IS', 'CF', 'FINA', 'ANALYST', 'MKLINE', 'EXPO'):
                stack.enter_context(patch.object(getattr(self.vendor, name), 'clear_all'))
            self.vendor.clear_all()
        self.assertIsNone(self.vendor._list_day_price)
        self.assertNotIn('all_stocks', self.vendor.__dict__)

    def test_invalid_listing_reference_stays_unknown(self):
        for value in (0., -1., np.inf, np.nan):
            with self.subTest(value=value), patch('src.data.loader.data_vendor.DB.load', return_value=self.day.assign(preclose=value)):
                self.assertTrue(self.vendor.get_list_day_price(20260106, 20260106).adj_list_price.isna().all())

    def test_period_close_includes_listing_at_or_after_start(self):
        with patch('src.data.loader.data_vendor.DB.load', return_value=self.day):
            listing = self.vendor.get_list_day_price(20260106, 20260106)
        for start in (20260105, 20260106):
            for adj_present in (True, False):
                with self.subTest(start=start, adj_present=adj_present):
                    # Existing stock 9 retains its ordinary interval performance.
                    p0 = pd.Series({9: 10., 1: 24.}, name='close')
                    p1 = pd.Series({9: 11., 1: 30., 2: 22.}, name='close')
                    with (patch.object(TRADE, '_adj_price_col', side_effect=[p0, p1] if adj_present else [None, None]),
                          patch.object(TRADE, '_period_return_from_day', return_value=pd.DataFrame({'ret': [.1]}, index=pd.Index([9], name='secid'))),
                          patch.object(TRADE, 'get_trd', return_value=pd.DataFrame({'secid': [9, 1, 2], 'close': [11., 15., 22.], 'adjfactor': [1., 2., 1.]})),
                          patch.object(DATAVENDOR, 'get_list_day_price', return_value=listing)):
                        result: Any = TRADE.period_return(start, 20260107)
                    self.assertAlmostEqual(result.loc[9, 'ret'], .1)
                    self.assertAlmostEqual(result.loc[1, 'ret'], .5)
                    self.assertAlmostEqual(result.loc[2, 'ret'], .1)

    def test_period_open_vwap_never_use_listing_reference(self):
        for price_type in ('open', 'vwap'):
            with (patch.object(TRADE, '_adj_price_col', side_effect=[pd.Series({1: 12.}), pd.Series({1: 15.})]),
                  patch.object(DATAVENDOR, 'get_list_day_price') as listing):
                result: Any = TRADE.period_return(20260106, 20260107, price_type)
            self.assertAlmostEqual(result.loc[1, 'ret'], .25)
            listing.assert_not_called()

    def test_period_after_listing_and_missing_endpoint(self):
        with (patch.object(TRADE, '_adj_price_col', side_effect=[pd.Series({1: 12.}), pd.Series({1: 15.})]),
              patch.object(DATAVENDOR, 'get_list_day_price', return_value=pd.DataFrame())):
            result: Any = TRADE.period_return(20260107, 20260108)
            self.assertAlmostEqual(result.loc[1, 'ret'], .25)
        with patch('src.data.loader.data_vendor.DB.load', return_value=self.day):
            listing = self.vendor.get_list_day_price(20260106, 20260106)
        with (patch.object(TRADE, '_adj_price_col', side_effect=[pd.Series({9: 10.}), pd.Series({9: 11.})]),
              patch.object(DATAVENDOR, 'get_list_day_price', return_value=listing)):
            result: Any = TRADE.period_return(20260105, 20260107)
        self.assertTrue(np.isnan(result.loc[1, 'ret']))

    def test_daily_block_fills_only_listing_close_and_keeps_vwap(self):
        frame = pd.DataFrame({'secid': [1, 1, 1], 'date': [20260105, 20260106, 20260107],
                              'close': [np.nan, 24., 30.], 'vwap': [np.nan, 22., 28.]})
        block = DataBlock.from_pandas(frame)
        dates = np.array([20260105, 20260106, 20260107])
        with (patch.object(self.vendor, 'td_within', return_value=dates),
              patch.object(self.vendor, 'get_quotes_block', return_value=block),
              patch('src.data.loader.data_vendor.DB.load', return_value=self.day)):
            self.vendor.update_return_block(20260105, 20260107)
        out: Any = self.vendor.blocks_cache['daily_ret'].to_dataframe()
        self.assertAlmostEqual(out.loc[(1, 20260106), 'close'], .2, places=6)
        self.assertTrue(np.isnan(out.loc[(1, 20260106), 'vwap']))
        self.assertAlmostEqual(out.loc[(1, 20260107), 'close'], .25, places=6)
        self.assertAlmostEqual(out.loc[(1, 20260107), 'vwap'], 28/22-1, places=6)
