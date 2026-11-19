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
from src.data.maintenance.repair_microcap import check_prices

__all__ = ['AdjpriceCoverageTest']

class AdjpriceCoverageTest(unittest.TestCase):
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
        with (patch.object(adjprice, 'listing_description', return_value=pd.DataFrame(columns=['secid', 'list_dt', 'delist_dt'])),
              patch.object(adjprice, 'ensure_adjprice') as update):
            result = TradeDataAccess.get_adjprice(fake, 20261029, ['secid', 'close'])
            TradeDataAccess.get_adjprice(fake, 20261029)
        self.assertEqual(result.close.tolist(), [20., 40.])
        self.assertEqual(update.call_count, 2)
        update.assert_called_with(20261029, reason='query')
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
