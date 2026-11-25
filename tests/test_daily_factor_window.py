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
                models.update.assert_called_once_with()
                for name in ('market_factors', 'stock_factors', 'affiliate_factors', 'pooling_factors', 'factor_stats'):
                    self.assertEqual(getattr(api, 'update_' + name).call_args.kwargs['missing_since'], expected)

    def test_daily_predictions_do_not_scan_or_fit_production_models(self):
        import src.api.pkgs.model as api

        with (
            patch.object(api, 'wrap_update', side_effect=lambda fn, title: fn()),
            patch.object(api.ModelAPI, 'prepare_predict_data') as prepare,
            patch.object(api.ArchivedPredictorModel, 'update') as predict,
            patch.object(api.ModelPortfolioBuilder, 'update') as portfolio,
            patch.object(api.ModelTrainer, 'update_models') as fit,
        ):
            api.ModelAPI.update()
        prepare.assert_called_once()
        predict.assert_called_once()
        portfolio.assert_called_once()
        fit.assert_not_called()

    def test_production_batch_scans_once_and_continues_after_model_failure(self):
        from src.res.model.model_module.application import trainer
        from src.res.model.util.lifecycle import ModelLifecycle
        from src.proj.util.script.fit_lock import FitLock
        from contextlib import contextmanager

        events = []

        @contextmanager
        def wait_for_fit(*, on_wait):
            on_wait(0.)
            on_wait(300.)
            yield 305.

        with (
            patch.object(trainer, 'MACHINE', SimpleNamespace(cuda_server=True)),
            patch.object(trainer, 'Proj'),
            patch.object(ModelLifecycle, 'inspect_registered', return_value=[
                {'base': '/tmp/day', 'name': 'day'},
                {'base': '/tmp/week', 'name': 'week'},
            ]) as inspect,
            patch.object(ModelLifecycle, 'repair', autospec=True, side_effect=[ValueError('bad weights'), None]) as resume,
            patch.object(FitLock, 'production_guard', side_effect=wait_for_fit) as priority_fit,
        ):
            def notify(kind, message):
                events.append((kind, message))

            with self.assertRaises(ExceptionGroup):
                trainer.ModelTrainer.update_models(production_notice=notify)
            inspect.assert_called_once()
            self.assertEqual([call.args[1]['name'] for call in resume.call_args_list], ['day', 'week'])
            self.assertEqual([kind for kind, _ in events], ['waiting', 'training', 'finished'])
            self.assertIn('305.0s', events[1][1])
            self.assertIn('Successful models: week', events[-1][1])
            self.assertIn('day: ValueError: bad weights', events[-1][1])
            priority_fit.assert_called_once()
            events.clear()
            inspect.return_value = []
            trainer.ModelTrainer.update_models(production_notice=notify)
            self.assertEqual(events, [])
            priority_fit.assert_called_once()

            def scan_error(**kwargs):
                kwargs['on_error']('bad-model', ValueError('invalid config'))
                return []

            inspect.side_effect = scan_error
            with self.assertRaises(ExceptionGroup):
                trainer.ModelTrainer.update_models(production_notice=notify)
            self.assertEqual([kind for kind, _ in events], ['finished'])
            self.assertIn('bad-model: ValueError: invalid config', events[0][1])
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
