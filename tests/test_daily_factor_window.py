from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.api.pkgs import update
from src.api.pkgs import factor
from src.res.factor.calculator.updater import BaseFactorUpdater
from src.proj import Base


class DailyFactorWindowTest(unittest.TestCase):
    def test_daily_passes_two_calendar_year_window_to_every_factor_stage(self):
        for target, expected in [(20261116, 20241116), (20240229, 20220228)]:
            with self.subTest(target=target), ExitStack() as stack:
                stack.enter_context(patch.object(update, 'MACHINE', SimpleNamespace(updatable=True)))
                stack.enter_context(patch.object(update.CALENDAR, 'update_to', return_value=target))
                stack.enter_context(patch.object(update.DiskTTLCache, 'get', return_value=SimpleNamespace(valid_value=target)))
                for name in ('DataAPI', 'ModelAPI', 'TradingAPI', 'SummaryAPI', 'NotificationAPI', 'Proj', 'print_update_records'):
                    stack.enter_context(patch.object(update, name))
                stack.enter_context(patch('src.res.model.util.lifecycle.ModelLifecycle.inspect_registered'))
                api = stack.enter_context(patch.object(update, 'FactorAPI'))
                update.UpdateAPI.daily()
                for name in ('market_factors', 'stock_factors', 'affiliate_factors', 'pooling_factors', 'factor_stats'):
                    self.assertEqual(getattr(api, 'update_' + name).call_args.kwargs['missing_since'], expected)

    def test_factor_apis_forward_window(self):
        with patch.object(factor, 'wrap_update') as wrap:
            for name in ('market_factors', 'stock_factors', 'affiliate_factors', 'pooling_factors', 'factor_stats'):
                getattr(factor.FactorAPI, 'update_' + name)(missing_since=20241116)
                self.assertEqual(wrap.call_args.kwargs['missing_since'], 20241116)

    def test_window_clamps_automatic_range_only(self):
        with patch.object(Base.BasicUpdater, 'parse_update_input', return_value=dict(start=20110101, end=20261116, overwrite=False)):
            result = BaseFactorUpdater.parse_update_input(missing_since=20241116)
            self.assertEqual(result['start'], 20241116)
            self.assertFalse(result['overwrite'])
            self.assertEqual(BaseFactorUpdater.parse_update_input()['start'], 20110101)
        with patch.object(Base.BasicUpdater, 'parse_update_input', return_value=dict(start=20110101, end=20261116, overwrite=True)):
            self.assertEqual(BaseFactorUpdater.parse_update_input(missing_since=20241116)['start'], 20110101)
