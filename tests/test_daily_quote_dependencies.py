"""Derived prices and indices must not turn a missing market day into valid data."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from src.data.update.custom import adjprice, custom_index
from src.proj import Dates

__all__ = ['DailyQuoteDependenciesTest']


class DailyQuoteDependenciesTest(unittest.TestCase):
    def test_nonempty_rebalance_calendar_builds_actual_portfolio(self):
        index = custom_index.MicroCap_400()
        index.reb_dates = Dates([20261117, 20261118])
        weights = pd.DataFrame({'secid': [1], 'weight': [1.0]})
        with patch.object(index, 'rebalance_portfolio', return_value=weights) as rebalance:
            portfolio = index.index_portfolio(20261117)
        rebalance.assert_called_once_with(20261117)
        self.assertFalse(portfolio.emtpy)

    def test_first_microcap_return_has_previous_close_holdings(self):
        from src.data.preprocess.repair_microcap import checked_return

        day = pd.DataFrame({'secid': [1, 2], 'close': [10.0, 20.0], 'adjfactor': [1.0, 1.0]})
        weights = pd.DataFrame({'secid': [1, 2], 'weight': [0.4, 0.6]})
        returns = pd.DataFrame({'secid': [1, 2], 'ret': [0.01, -0.02]}).set_index('secid')
        # Cover a holiday start and an actual trading-day start.
        for start in (20050101, 20050104):
            index = custom_index.MicroCap_400()
            index.START_DATE = start
            with (self.subTest(start=start),
                  patch.object(custom_index.CALENDAR, 'updated', return_value=20050105),
                  patch.object(custom_index.DB, 'load', return_value=day),
                  patch.object(index, 'rebalance_portfolio', return_value=weights) as rebalance,
                  patch.object(custom_index.DATAVENDOR, 'get_quote_ret', return_value=returns)):
                self.assertEqual(index.reb_dates.min, 20041231)
                self.assertTrue(index.index_portfolio(20041230).emtpy)
                for date, previous in ((20050104, 20041231), (20050105, 20050104)):
                    value = index.index_return(date) * 100
                    self.assertAlmostEqual(value, checked_return(weights, returns, date))
                    self.assertAlmostEqual(value, -0.8)
                    self.assertEqual(rebalance.call_args.args[0], previous)
                # The seed holding date does not extend the exported return history.
                self.assertEqual(index.target_dates(overwrite=True).min, 20050104)

    def test_microcap_count_uses_half_listed_market_before_st_exclusion(self):
        for population, st_count in ((1000, 0), (800, 0), (799, 0), (101, 10), (100, 70)):
            with self.subTest(population=population, st_count=st_count):
                secids = np.arange(population)
                valuation = pd.DataFrame({'secid': secids, 'free_share': 1., 'close': secids + 1.})
                info = custom_index.DATAVENDOR.INFO
                with (patch.object(custom_index.DATAVENDOR.TRADE, 'get_val', return_value=valuation),
                      patch.object(custom_index.DATAVENDOR.TRADE, 'get_trd', return_value=valuation.assign(status=1)),
                      patch.object(info, 'get_desc', return_value=pd.DataFrame({'secid': secids})) as description,
                      patch.object(info, 'get_st', return_value=pd.DataFrame({'secid': secids[:st_count]}))):
                    weights = custom_index.MicroCap_400().rebalance_portfolio(20041231)
                count = min(400, population // 2, population - st_count)
                self.assertEqual(len(weights), count)
                np.testing.assert_array_equal(weights.secid, secids[st_count:st_count + count])
                np.testing.assert_allclose(weights.weight, 1 / count)
                self.assertAlmostEqual(weights.weight.sum(), 1.)
                self.assertEqual(description.call_args.kwargs['exchange'], ['SZSE', 'SSE'])
                self.assertEqual(int(description.call_args.args[0]), 20041230)

    def test_microcap_excludes_suspended_unknown_and_absent_holdings_day_quotes(self):
        secids = np.arange(1, 7)
        valuation = pd.DataFrame({'secid': secids, 'free_share': 1., 'close': secids.astype(float)})
        trading = pd.DataFrame({'secid': [1, 2, 4, 5, 6], 'status': [0, np.nan, 1, 1, 1], 'close': 10.})
        with (patch.object(custom_index.DATAVENDOR.TRADE, 'get_val', return_value=valuation),
              patch.object(custom_index.DATAVENDOR.TRADE, 'get_trd', return_value=trading) as quotes,
              patch.object(custom_index.DATAVENDOR.INFO, 'get_desc', return_value=pd.DataFrame({'secid': secids})),
              patch.object(custom_index.DATAVENDOR.INFO, 'get_st', return_value=pd.DataFrame({'secid': []}))):
            weights = custom_index.MicroCap_400().rebalance_portfolio(20041231)
        np.testing.assert_array_equal(weights.secid, [4, 5, 6])
        quotes.assert_called_once_with(20041231)

    def test_microcap_start_and_repair_default_are_2005(self):
        import inspect
        from src.data.preprocess.repair_microcap import rebuild_microcap

        self.assertEqual(custom_index.MicroCap_400.START_DATE, 20050101)
        self.assertEqual(inspect.signature(rebuild_microcap).parameters['start'].default, 20050101)

    def test_missing_or_invalid_day_never_saves_adjprice(self):
        for day in (None, pd.DataFrame(), pd.DataFrame({'secid': [1]})):
            with self.subTest(day=repr(day)), patch.object(adjprice.DB, 'load', return_value=day), patch.object(adjprice.DB, 'save') as save:
                with self.assertRaisesRegex(ValueError, 'adjprice at 20261118'):
                    adjprice.AdjPriceUpdater.update_one(20261118)
                save.assert_not_called()

    def test_valid_day_still_fills_individual_halted_stock(self):
        day = pd.DataFrame({'secid': [1], 'adjfactor': [2.0], **{c: [10.0] for c in adjprice.PRICE_COLS}})
        previous = pd.DataFrame({'secid': [1, 2], **{c: [19.0, 30.0] for c in adjprice.PRICE_COLS}})
        with patch.object(adjprice.DB, 'load', side_effect=[day, previous]), patch.object(adjprice.INFO, 'get_secid', return_value=np.array([1, 2])), patch.object(adjprice.CALENDAR, 'td', return_value=SimpleNamespace(as_int=lambda: 20261117)):
            result = adjprice.calc_adjprice(20261118).set_index('secid')
        self.assertEqual(result.loc[1, 'close'], 20.0)
        self.assertEqual(result.loc[2, 'close'], 30.0)

    def test_missing_day_never_calculates_or_saves_index(self):
        for day in (None, pd.DataFrame(), pd.DataFrame({'secid': [1]})):
            index = custom_index.MicroCap_400()
            with self.subTest(day=repr(day)), patch.object(custom_index.DB, 'load', return_value=day), patch.object(custom_index.CALENDAR, 'updated', return_value=20261118), patch.object(index, 'index_portfolio') as portfolio, patch.object(custom_index.Save, 'df') as save:
                with self.assertRaisesRegex(ValueError, 'microcap_400 at 20261118'):
                    index.update_dates([20261118])
                portfolio.assert_not_called()
                save.assert_not_called()

    def test_valid_day_keeps_index_return_calculation(self):
        day = pd.DataFrame({'secid': [1], 'close': [10.0], 'adjfactor': [1.0]})
        portfolio = Mock(emtpy=False)
        portfolio.fut_ret.return_value = 0.025
        index = custom_index.MicroCap_400()
        with patch.object(custom_index.DB, 'load', return_value=day), patch.object(custom_index.CALENDAR, 'td', return_value=SimpleNamespace(td=20261117)), patch.object(index, 'index_portfolio', return_value=portfolio):
            self.assertEqual(index.index_return(20261118), 0.025)
        portfolio.fut_ret.assert_called_once_with(20261118)
