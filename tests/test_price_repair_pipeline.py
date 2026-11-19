"""Historical price prerequisites precede microcap and market regeneration."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.proj import Dates
from src.data.update.custom import listing_price, adjprice
from src.data.preprocess import repair_microcap as repair

__all__ = ['PriceRepairPipelineTest']

class PriceRepairPipelineTest(unittest.TestCase):
    def test_listing_build_refresh_and_missing_reference(self):
        desc = pd.DataFrame({'secid': [1, 2], 'list_dt': [20041230, 20041230]})
        quotes = pd.DataFrame({'secid': [1, 2], 'preclose': [10., np.nan], 'adjfactor': [2., 1.]})
        with (patch.object(listing_price.DB, 'load', side_effect=[desc, quotes]),
              patch.object(listing_price.DB, 'dates', return_value=Dates([20041230]))):
            table = listing_price.build_listing_prices(rebuild=True).set_index('secid')
        self.assertEqual(table.loc[1, 'adj_list_price'], 20.)
        self.assertTrue(pd.isna(table.loc[2, 'adj_list_price']))

    def test_halt_fill_before_2005_uses_previous_staged_panel(self):
        day = pd.DataFrame({'secid': [1], 'adjfactor': [1.], **{c: [10.] for c in adjprice.PRICE_COLS}})
        previous = pd.DataFrame({'secid': [1, 2], **{c: [9., 20.] for c in adjprice.PRICE_COLS}})
        with (patch.object(adjprice.DB, 'load', return_value=day) as load,
              patch.object(adjprice.INFO, 'get_desc', return_value=pd.DataFrame({'secid': [1, 2], 'list_dt': [20000101]*2, 'delist_dt': [99991231]*2}))):
            result = adjprice.calc_adjprice(20041231, previous=previous).set_index('secid')
        self.assertEqual(result.loc[2, 'close'], 20.)
        self.assertEqual(load.call_count, 1)

    def test_upstream_stages_before_install_and_dry_run_never_installs(self):
        for dry in (False, True):
            with self.subTest(dry=dry), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                candidate = pd.DataFrame({'secid': [1], 'close': [10.]})
                with (patch.object(repair, 'PATH', SimpleNamespace(result=root)),
                      patch.object(repair.DB, 'dates', return_value=Dates([20041230, 20041231])),
                      patch.object(repair.DB, 'path', side_effect=lambda source, key, *date: root / f'{key}{date}.feather'),
                      patch.object(adjprice, 'calc_adjprice', return_value=candidate) as calc,
                      patch.object(listing_price, 'build_listing_prices', return_value=pd.DataFrame({'secid': [1], 'list_dt': [20041230], 'adj_list_price': [10.]})),
                      patch.object(repair, 'install') as install):
                    if dry:
                        with self.assertRaisesRegex(ValueError, 'no live files changed'):
                            repair.repair_price_inputs(True)
                        install.assert_not_called()
                    else:
                        result = repair.repair_price_inputs()
                        self.assertEqual(result['changed_files'], 3)
                        install.assert_called_once()
                    self.assertTrue(calc.call_args_list[0].kwargs['previous'].empty)
                    pd.testing.assert_frame_equal(calc.call_args_list[1].kwargs['previous'], candidate)

    def test_persistent_daily_update_replaces_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'list_day_price.feather'
            old = pd.DataFrame({'secid': [1], 'list_price': [9.]})
            old.to_feather(path)
            updated = pd.DataFrame({'secid': [1], 'list_price': [10.]})
            with (patch.object(listing_price.DB, 'path', return_value=path),
                  patch.object(listing_price, 'build_listing_prices', return_value=updated) as build):
                listing_price.update_listing_prices(20041230)
            build.assert_called_once_with(refresh_date=20041230)
            pd.testing.assert_frame_equal(pd.read_feather(path), updated)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_daily_quote_hook_order(self):
        from typing import Any
        from src.data.download.tushare.task.t01_daily import DailyQuote
        events = []
        fake: Any = SimpleNamespace(check_server_down=lambda: False,
            get_data=lambda date: pd.DataFrame({'secid': [1]}), DB_SRC='trade_ts', DB_KEY='day', indent=0, vb_level=0)
        with (patch.object(listing_price.DB, 'save', side_effect=lambda *a, **kw: events.append('day')),
              patch.object(adjprice.AdjPriceUpdater, 'update_one', side_effect=lambda date: events.append('adjprice')),
              patch.object(listing_price, 'update_listing_prices', side_effect=lambda date: events.append('listing'))):
            DailyQuote.update_dates(fake, [20041230])
        self.assertEqual(events, ['day', 'adjprice', 'listing'])

    def test_pipeline_order_and_failure_stops_downstream(self):
        events = []
        with (patch.object(repair, 'repair_price_inputs', side_effect=lambda **kw: events.append('prices')),
              patch.object(repair.DATAVENDOR, 'clear_all', side_effect=lambda: events.append('clear')),
              patch.object(repair, 'rebuild_microcap', side_effect=lambda *a: events.append('index'))):
            repair.repair_all(20050101, 20050104)
        self.assertEqual(events, ['prices', 'clear', 'index'])
        with (patch.object(repair, 'repair_price_inputs', side_effect=ValueError('bad prices')),
              patch.object(repair, 'rebuild_microcap') as downstream):
            with self.assertRaisesRegex(ValueError, 'bad prices'):
                repair.repair_all(20050101, 20050104)
            downstream.assert_not_called()
