"""Coverage repairs preserve halted/unknown names and respect lifecycle bounds."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from typing import Any
import numpy as np
import pandas as pd
from src.data.util.prices import adjusted as adjprice
from src.data.update.custom import custom_index
from src.data.loader.trade_data import TradeDataAccess
from src.data.loader.data_vendor import DataVendor
from src.data.maintenance.repair_microcap import check_prices

__all__ = ['AdjpriceCoverageTest']

class AdjpriceCoverageTest(unittest.TestCase):
    def test_query_repair_requires_recent_known_listing(self):
        frame = pd.DataFrame({'secid': [1], 'close': [10.]})
        fake: Any = SimpleNamespace(get=Mock(return_value=frame), get_adjprice=Mock(return_value=frame))
        # Age is measured in calendar days at the requested price date.
        for listed, eligible in ((20250101, False), (20250102, True),
                                 (20260101, True), (20260102, False), (0, False)):
            desc = pd.DataFrame({'secid': [2], 'list_dt': [listed], 'delist_dt': [99991231]})
            with (self.subTest(listed=listed),
                  patch.object(adjprice, 'listing_description', return_value=desc),
                  patch.object(adjprice, 'ensure_adjprice') as ensure,
                  patch('src.data.loader.trade_data.Logger.warning')):
                TradeDataAccess.get_adjprice_with_ensure(fake, 20260101, [1, 2, 3])
                self.assertEqual(ensure.call_count, int(eligible))
                self.assertNotIn(3, adjprice.recent_listing_ids(20260101, [3]))

    def test_frozen_prices_bypass_live_cache_in_both_directions(self):
        trade = object.__new__(TradeDataAccess)
        live = pd.DataFrame({'secid': [1], 'close': [20.]})
        frozen = live.assign(close=10.)
        with (patch.object(trade, 'get', return_value=live) as cached,
              patch.object(trade, 'data_loader', return_value=frozen),
              patch('src.data.loader.trade_data.snapshot_active', return_value=True)):
            self.assertEqual(trade.get_adjprice(20050104, 'close').iloc[0, 0], 10.)
            self.assertEqual(trade.get_adjprice_with_ensure(20050104, [1]).close.item(), 10.)
            cached.assert_not_called()

    def test_plain_read_and_complete_holdings_do_not_ensure(self):
        frame = pd.DataFrame({'secid': [1], 'close': [10.]})
        fake: Any = SimpleNamespace(get=Mock(return_value=frame))
        fake.get_adjprice = lambda date: TradeDataAccess.get_adjprice(fake, date)
        with patch.object(adjprice, 'ensure_adjprice') as ensure:
            TradeDataAccess.get_adjprice(fake, 20050104)
            TradeDataAccess.get_adjprice_with_ensure(fake, 20050104, [1])
        ensure.assert_not_called()
        self.assertFalse(any(c.kwargs.get('overwrite') for c in fake.get.call_args_list))

    def test_unresolved_holdings_warn_and_portfolio_returns_zero(self):
        frame = pd.DataFrame({'secid': [1], 'close': [10.], 'open': [11.]})
        trade: Any = SimpleNamespace(get=Mock(return_value=frame))
        trade.get_adjprice = lambda date: TradeDataAccess.get_adjprice(trade, date)
        trade.get_adjprice_with_ensure = lambda *a: TradeDataAccess.get_adjprice_with_ensure(trade, *a)
        vendor: Any = SimpleNamespace(TRADE=trade)
        for failure in (None, ValueError('missing source')):
            for price in ('close', 'open'):
                with (self.subTest(failure=failure, price=price),
                      patch.object(adjprice, 'recent_listing_ids', return_value={2}),
                      patch.object(adjprice, 'ensure_adjprice', side_effect=failure),
                      patch('src.data.loader.trade_data.Logger.warning') as warning):
                    result = DataVendor.get_quote_ret(vendor, 20041231, 20050104, 'close', price, [1, 2])
                    self.assertEqual(result.index.tolist(), [1, 2])
                    self.assertEqual(result.loc[2, 'ret'], 0.)
                    np.testing.assert_allclose(result.loc[[1], 'ret'].to_numpy(), [0. if price == 'close' else .1])
                    self.assertTrue(any('secid=[2]' in str(c) for c in warning.call_args_list))

    def test_precomputed_portfolio_quotes_use_held_prices_on_listing_day(self):
        frames = {
            20260106: pd.DataFrame({'secid': [1, 2], 'close': [12., 30.]}),
            20260107: pd.DataFrame({'secid': [1, 3], 'close': [15., 10.]}),
        }
        trade = Mock()
        trade.get_adjprice.side_effect = lambda date, fields: frames[date][fields]
        trade.get_adjprice_with_ensure.side_effect = lambda date, ids, fields: frames[date][fields]
        vendor: Any = SimpleNamespace(TRADE=trade)
        panel = DataVendor.get_quote_ret(vendor, 20260106, 20260107, portfolio=True)
        held = DataVendor.get_quote_ret(vendor, 20260106, 20260107, secid=[1, 2, 3])
        self.assertEqual(panel.ret.to_dict(), {1: .25})
        self.assertEqual(held.ret.to_dict(), {1: .25, 2: 0., 3: 0.})
        trade.period_return.assert_not_called()

    def quotes(self, ids, prices):
        return pd.DataFrame({'secid': ids, 'adjfactor': 2., **{c: prices for c in adjprice.PRICE_COLS}})

    def test_multiple_listing_intervals_writer_coverage_and_repair_agree(self):
        desc = pd.DataFrame({'secid': [600018, 600018],
                             'list_dt': [20061026, 20000719],
                             'delist_dt': [99991231, 20061020]})
        day = self.quotes([600018, 999999], [7.62, 10.])
        previous = day.drop(columns='adjfactor')
        for date, included in ((20000718, False), (20000719, True),
                               (20041231, True), (20061019, True),
                               (20061020, False), (20061025, False),
                               (20061026, True), (20261118, True)):
            with (self.subTest(date=date),
                  patch.object(adjprice, 'listing_description', return_value=desc),
                  patch.object(adjprice.DB, 'load', return_value=day)):
                result = adjprice.calc_adjprice(date, previous)
                self.assertEqual(600018 in result.secid.values, included)
                self.assertIn(999999, result.secid.values)
                self.assertTrue(adjprice.covers_day(result, day, date))
                if included:
                    np.testing.assert_allclose(result.loc[result.secid.eq(600018), 'close'].to_numpy(dtype=float), [15.24])
                    self.assertFalse(adjprice.covers_day(result[result.secid != 600018], day, date))
                with patch.object(adjprice.DB, 'load', side_effect=[day, result]):
                    check_prices(date)

    def test_unknown_quote_and_halted_previous_survive_but_inactive_removed(self):
        day = self.quotes([1, 920306, 3, 4], [10., 9.88, 5., 6.])
        previous = self.quotes([2, 5, 6], [20., 30., 40.]).drop(columns='adjfactor')
        desc = pd.DataFrame({'secid': [1, 2, 3, 4, 6], 'list_dt': [20000101, 20000101, 20261101, 20000101, 20000101],
                             'delist_dt': [99991231, 99991231, 99991231, 20261029, 20261028]})
        with (patch.object(adjprice.DB, 'load', return_value=day),
              patch.object(adjprice, 'listing_description', return_value=desc)):
            result = adjprice.calc_adjprice(20261029, previous).set_index('secid')
        self.assertEqual(set(result.index), {1, 2, 5, 920306})
        self.assertEqual(result.loc[2, 'close'], 20.)
        self.assertEqual(result.loc[920306, 'close'], 19.76)

    def test_coverage_missing_cells_and_intentional_exclusions(self):
        day = self.quotes([1, 2], [10., 20.])
        desc = pd.DataFrame({'secid': [2], 'list_dt': [20270101], 'delist_dt': [99991231]})
        good = self.quotes([1], [20.])
        with patch.object(adjprice, 'listing_description', return_value=desc):
            self.assertTrue(adjprice.covers_day(good, day, 20261029))
            self.assertFalse(adjprice.covers_day(good.assign(close=np.nan), day, 20261029))
            self.assertFalse(adjprice.covers_day(pd.DataFrame(), day, 20261029))

    def test_get_repairs_and_reloads_only_when_incomplete(self):
        day = self.quotes([1, 2], [10., 20.])
        old = self.quotes([1], [20.])
        new = self.quotes([1, 2], [20., 40.])
        state = {'frame': old}
        def get(date, kind, field=None, overwrite=False):
            if overwrite:
                state['frame'] = new
            return state['frame'] if field is None else state['frame'][field]
        fake: Any = SimpleNamespace(get=Mock(side_effect=get), get_trd=Mock(return_value=day))
        fake.get_adjprice = lambda date: TradeDataAccess.get_adjprice(fake, date)
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=['secid', 'list_dt', 'delist_dt'])),
              patch.object(adjprice, 'recent_listing_ids', return_value={2}),
              patch.object(adjprice, 'ensure_adjprice') as update):
            result = TradeDataAccess.get_adjprice_with_ensure(fake, 20261029, [1, 2], ['secid', 'close'])
            TradeDataAccess.get_adjprice(fake, 20261029)
        self.assertEqual(result.close.tolist(), [20., 40.])
        self.assertEqual(update.call_count, 1)
        update.assert_called_with(20261029, force=True, reason='holdings-query')
        self.assertTrue(any(c.kwargs.get('overwrite') for c in fake.get.call_args_list))

    def test_writer_replaces_atomically_and_preserves_original_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adjprice.feather'
            original = self.quotes([1], [10.])
            original.to_feather(path)
            new = self.quotes([1, 2], [20., 40.])
            with (patch.object(adjprice.DB, 'path', return_value=path),
                  patch.object(adjprice, 'calc_adjprice', return_value=new),
                  patch.object(adjprice.os, 'replace', side_effect=OSError('write failed'))):
                with self.assertRaisesRegex(OSError, 'write failed'):
                    adjprice.atomic_frame(new, path)
            pd.testing.assert_frame_equal(pd.read_feather(path), original)
            with (patch.object(adjprice.DB, 'path', return_value=path),
                  patch.object(adjprice, 'calc_adjprice', return_value=new)):
                adjprice.atomic_frame(new, path)
            pd.testing.assert_frame_equal(pd.read_feather(path), new)

    def test_microcap_uses_exchange_filtered_universe(self):
        ids = [1, 2, 3, 4, 920306, 830001, 430001]
        valuation = pd.DataFrame({'secid': ids, 'free_share': 1., 'close': [10., 20., 30., 40., 1., 2., 3.]})
        with (patch.object(custom_index.DATAVENDOR.TRADE, 'get_val', return_value=valuation),
              patch.object(custom_index.DATAVENDOR.INFO, 'get_desc', return_value=pd.DataFrame({'secid': ids[:4]})) as desc,
              patch.object(custom_index.DATAVENDOR.INFO, 'get_st', return_value=pd.DataFrame({'secid': []}))):
            weights = custom_index.MicroCap_400().rebalance_portfolio(20261029)
        self.assertEqual(weights.secid.tolist(), [1, 2])
        self.assertEqual(desc.call_args.kwargs['exchange'], ['SZSE', 'SSE'])
