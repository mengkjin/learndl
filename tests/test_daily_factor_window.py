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
                models = stack.enter_context(patch.object(update, 'ModelAPI'))
                for name in ('DataAPI', 'TradingAPI', 'SummaryAPI', 'NotificationAPI', 'Proj', 'print_update_records'):
                    stack.enter_context(patch.object(update, name))

                api = stack.enter_context(patch.object(update, 'FactorAPI'))
                update.UpdateAPI.daily()
                models.update.assert_called_once_with(update_due_models=True)
                for name in ('market_factors', 'stock_factors', 'affiliate_factors', 'pooling_factors', 'factor_stats'):
                    self.assertEqual(getattr(api, 'update_' + name).call_args.kwargs['missing_since'], expected)

    def test_daily_checks_and_runs_due_models_before_predictions(self):
        import src.api.pkgs.model as api
        from src.res.model.model_module.application import trainer
        from src.res.model.util.lifecycle import ModelLifecycle
        from src.proj.util.script.fit_lock import FitLock
        from contextlib import nullcontext

        events = []
        with (
            patch.object(api, 'MACHINE', SimpleNamespace(cuda_server=True)),
            patch.object(trainer, 'MACHINE', SimpleNamespace(cuda_server=True)),
            patch.object(trainer, 'Proj'),
            patch.object(api, 'wrap_update', side_effect=lambda fn, title, **kwargs: fn(**kwargs)),
            patch.object(api.ModelAPI, 'prepare_predict_data', side_effect=lambda: events.append('data')),
            patch.object(api.ArchivedPredictorModel, 'update', side_effect=lambda: events.append('predict')),
            patch.object(api.ModelPortfolioBuilder, 'update', side_effect=lambda: events.append('portfolio')),
            patch.object(ModelLifecycle, 'inspect_registered',
                         side_effect=lambda **kwargs: events.append('check') or [{'base': '/tmp/day'}]) as inspect,
            patch.object(ModelLifecycle, 'repair', autospec=True, side_effect=lambda *args: events.append('fit')) as resume,
            patch.object(FitLock, 'production_guard', return_value=nullcontext()) as priority_fit,
        ):
            api.ModelAPI.update(update_due_models=True)
            self.assertEqual(events, ['data', 'check', 'fit', 'predict', 'portfolio'])
            manager, request, run = resume.call_args.args
            self.assertEqual(str(manager.base), request['base'])
            self.assertEqual(run, trainer.ModelTrainer.GO)
            priority_fit.assert_called_once()
            events.clear()
            inspect.side_effect = lambda **kwargs: events.append('check') or []
            api.ModelAPI.update(update_due_models=True)
            self.assertEqual(events, ['data', 'check', 'predict', 'portfolio'])
            priority_fit.assert_called_once()

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
