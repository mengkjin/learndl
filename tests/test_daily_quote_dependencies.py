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
