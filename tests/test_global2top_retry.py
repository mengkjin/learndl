"""Global2Top retries must learn their own global phase before optimizing top."""
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import pandas as pd
import torch

from src.proj.bases import FittingEventType
from src.res.model.callback.fit.retrain import BadAttemptRetrain
from src.res.model.callback.specific.global2top import SpecificCB_Global2Top
from src.res.model.util.trainer.base_trainer import BaseTrainer


def make_callback(**kwargs):
    trainer = Mock(spec=BaseTrainer)
    trainer.is_fitting = True
    trainer.status = SimpleNamespace(
        epoch=0, epoch_key='Ph0 Ep0', status={}, add_epoch_event=Mock(),
    )
    trainer.metrics.batch_metrics.losses = {
        'global.ccc': torch.tensor(1.), 'top.soft_topk': torch.tensor(1.),
    }
    callback = SpecificCB_Global2Top(trainer, **kwargs)
    trainer.on_new_attempt.side_effect = lambda: callback.execute_hook('on_new_attempt')
    trainer.new_attempt.side_effect = lambda kind, **kw: BaseTrainer.new_attempt(trainer, kind, **kw)
    callback.on_fit_model_start()
    return trainer, callback


def set_valid_history(trainer, global_values, top_values):
    frame = pd.DataFrame(
        {'global.spearman': global_values, 'top.long_avg': top_values},
        index=pd.MultiIndex.from_product([[0], range(len(global_values))], names=['phase', 'epoch']),
    )
    trainer.metrics.attempt_metrics.tables = {'valid_epoch_accuracies': [frame]}


class Global2TopRetryTest(TestCase):
    def test_quality_retry_and_nan_redo_restart_global_without_erasing_model_history(self):
        for event in (FittingEventType.NEW_ATTEMPT, FittingEventType.REDO_ATTEMPT):
            with self.subTest(event=event):
                trainer, callback = make_callback()
                previous_best = object()
                trainer.metrics.model_metrics.attempt_metric_results = [previous_best]
                callback.fitting_phase = 'top'
                callback.global_plateaued = True
                callback.global_climax_epoch = 48
                callback.global_climax_level = .0315
                callback.global_support_level = .0215
                callback.valid_accuracies = pd.DataFrame({'old': [1.]})
                callback.loss_weights_set = callback.accuracy_computer_set = True

                BadAttemptRetrain(trainer).trigger_retrain(event, 'regression', new_lr_multiplier=.3)

                trainer.model.stack_model.assert_called_once()
                trainer.model.new_model.assert_called_once_with(lr_multiplier=.3)
                trainer.metrics.new_attempt.assert_called_once()
                self.assertEqual(callback.fitting_phase, 'global')
                self.assertFalse(callback.global_plateaued)
                self.assertEqual(callback.global_climax_epoch, -1)
                self.assertEqual(callback.global_climax_level, -1)
                self.assertEqual(callback.global_support_level, -1)
                self.assertTrue(callback.valid_accuracies.empty)
                trainer.metrics.set_accuracy_verdict.assert_called_once()
                self.assertEqual(trainer.metrics.model_metrics.attempt_metric_results, [previous_best])

                # Even beyond the old peak, the retry must not acquire top loss
                # or stop against the previous attempt's support level.
                trainer.status.add_epoch_event.reset_mock()
                for epoch in (0, 58, 69):
                    trainer.status.epoch = epoch
                    callback.on_fit_epoch_start()
                    callback.on_batch_metrics_after()
                    trainer.metrics.set_loss_weights.assert_called_with({'top.soft_topk': 0.})
                    aggregator = trainer.metrics.set_accuracy_aggregator.call_args.args[0]
                    self.assertEqual(aggregator({'global.spearman': .06, 'top.long_avg': -.1}), .06)
                    set_valid_history(trainer, [.01] * (epoch + 1), [0.] * (epoch + 1))
                    callback.collect_accuracies()
                    callback.check_top_converge()
                trainer.status.add_epoch_event.assert_not_called()

    def test_retry_can_reach_its_own_global_plateau_and_then_converge_in_top(self):
        trainer, callback = make_callback(protect=2, plateau_patience=2, converge_patience=2)
        callback.fitting_phase = 'top'
        callback.global_plateaued = True
        callback.global_climax_epoch = 48
        trainer.new_attempt('attempt')
        trainer.status.epoch = 2
        set_valid_history(trainer, [.2, .15, .1], [0., 0., 0.])
        trainer.metrics.attempt_metrics.best_epoch.return_value = SimpleNamespace(
            epoch=0, valid_accuracies={'global.spearman': .2, 'top.long_avg': 0.},
        )
        callback.on_validation_epoch_end()
        self.assertEqual(callback.fitting_phase, 'top')
        self.assertEqual(callback.global_climax_epoch, 0)
        self.assertEqual(callback.global_climax_level, .2)
        self.assertAlmostEqual(callback.global_support_level, .18)
        self.assertEqual(trainer.recall_ckpt.call_args.args, (0, 0))
        trainer.status.add_epoch_event.assert_not_called()

        callback.on_new_phase()  # A checkpoint recall must retain the global peak.
        self.assertEqual(callback.fitting_phase, 'top')
        trainer.status.epoch = 3
        trainer.status.epoch_key = 'Ph1 Ep3'
        callback.on_fit_epoch_start()
        callback.on_batch_metrics_after()
        self.assertGreater(trainer.metrics.set_loss_weights.call_args.args[0]['top.soft_topk'], 0.)
        aggregator = trainer.metrics.set_accuracy_aggregator.call_args.args[0]
        self.assertAlmostEqual(aggregator({'global.spearman': .2, 'top.long_avg': .3}), .3)
        frame = pd.DataFrame(
            {'global.spearman': [.2, .15, .1, .05], 'top.long_avg': [0., 0., 0., 0.]},
            index=pd.MultiIndex.from_tuples([(0, 0), (1, 1), (1, 2), (1, 3)], names=['phase', 'epoch']),
        )
        trainer.metrics.attempt_metrics.tables = {'valid_epoch_accuracies': [frame]}
        callback.on_validation_epoch_end()
        self.assertEqual(trainer.status.add_epoch_event.call_args.args[0], FittingEventType.END_ATTEMPT)
        trainer.recall_ckpt.assert_called_once()


if __name__ == '__main__':
    main()
