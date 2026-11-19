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
from src.data.update.custom import adjprice, custom_index
from src.data.loader.trade_data import TradeDataAccess

__all__ = ['AdjpriceCoverageTest']

class AdjpriceCoverageTest(unittest.TestCase):
    def quotes(self, ids, prices):
        return pd.DataFrame({'secid': ids, 'adjfactor': 2., **{c: prices for c in adjprice.PRICE_COLS}})

    def test_unknown_quote_and_halted_previous_survive_but_inactive_removed(self):
        day = self.quotes([1, 920306, 3, 4], [10., 9.88, 5., 6.])
        previous = self.quotes([2, 5, 6], [20., 30., 40.]).drop(columns='adjfactor')
        desc = pd.DataFrame({'secid': [1, 2, 3, 4, 6], 'list_dt': [20000101, 20000101, 20261101, 20000101, 20000101],
                             'delist_dt': [99991231, 99991231, 99991231, 20261029, 20261028]})
        with (patch.object(adjprice.DB, 'load', return_value=day),
              patch.object(adjprice.INFO, 'get_desc', return_value=desc)):
            result = adjprice.calc_adjprice(20261029, previous).set_index('secid')
        self.assertEqual(set(result.index), {1, 2, 5, 920306})
        self.assertEqual(result.loc[2, 'close'], 20.)
        self.assertEqual(result.loc[920306, 'close'], 19.76)

    def test_coverage_missing_cells_and_intentional_exclusions(self):
        day = self.quotes([1, 2], [10., 20.])
        desc = pd.DataFrame({'secid': [2], 'list_dt': [20270101], 'delist_dt': [99991231]})
        good = self.quotes([1], [20.])
        with patch.object(adjprice.INFO, 'get_desc', return_value=desc):
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
        with (patch.object(adjprice.INFO, 'get_desc', return_value=pd.DataFrame(columns=['secid', 'list_dt', 'delist_dt'])),
              patch.object(adjprice.AdjPriceUpdater, 'update_one') as update):
            result = TradeDataAccess.get_adjprice(fake, 20261029, ['secid', 'close'])
            TradeDataAccess.get_adjprice(fake, 20261029)
        self.assertEqual(result.close.tolist(), [20., 40.])
        update.assert_called_once_with(20261029)
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
                    adjprice.AdjPriceUpdater.update_one(20261029)
            pd.testing.assert_frame_equal(pd.read_feather(path), original)
            with (patch.object(adjprice.DB, 'path', return_value=path),
                  patch.object(adjprice, 'calc_adjprice', return_value=new)):
                adjprice.AdjPriceUpdater.update_one(20261029)
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
