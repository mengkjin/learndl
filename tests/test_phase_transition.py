"""Regression tests for logging boundaries, phase scheduling and capped weights."""
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from src.res.algo.nn.loss.loss import CappedSoftTopKLoss
from src.res.model.callback.monitor.summary import SummaryWriter
from src.res.model.util.trainer.base_trainer import BaseTrainer
from src.res.model.util.config import ModelConfig
from src.res.model.util.trainer.phase_transition import PhaseCyclicLR, capture_cycle


class SummaryLifecycleTest(TestCase):
    def make_writer(self):
        trainer = Mock(spec=BaseTrainer)
        trainer.status = SimpleNamespace(stage='fit', attempt_key='Trial0-0',
            model_num=0, model_date=20191217, model_epoch=40)
        trainer.fit_model_restart_count = 0
        trainer.config = Mock(spec=ModelConfig)
        trainer.config.base_path.model_clean_name = 'model'
        trainer.base_path.snapshot.side_effect = lambda *args: '/'.join(args)
        return trainer, SummaryWriter(trainer)

    @patch('src.res.model.callback.monitor.summary.TsboardWriter')
    def test_retry_tail_stays_old_then_identity_switches(self, factory):
        trainer, callback = self.make_writer()
        old = cast(Mock, callback.writer)
        callback.on_new_attempt()
        self.assertIs(callback.writer, old)
        old.close.assert_not_called()
        trainer.status.attempt_key = 'Trial1-0'
        _ = callback.writer
        old.close.assert_called_once()
        self.assertEqual(factory.call_count, 2)
        self.assertIn('Trial1-0', str(factory.call_args))
        trainer.status.attempt_key = 'Trial1-1'
        _ = callback.writer
        self.assertEqual(factory.call_count, 3)
        trainer.status.model_date = 20200101
        trainer.model_date = 20200101
        _ = callback.writer
        self.assertEqual(factory.call_count, 4)

    @patch('src.res.model.callback.monitor.summary.TsboardWriter')
    def test_restart_marks_existing_writer_before_switch_and_closes_at_end(self, factory):
        trainer, callback = self.make_writer()
        old = cast(Mock, callback.writer)
        trainer.fit_model_restart_count = 1
        callback.on_fit_model_restart()
        old.add_text.assert_called_once()
        old.close.assert_called_once()
        self.assertEqual(factory.call_count, 1)
        _ = callback.writer
        self.assertIn('ac_restart1', str(factory.call_args))
        callback.on_fit_model_end()
        self.assertNotIn('_writer', callback.__dict__)

    @patch('src.res.model.callback.monitor.summary.TsboardWriter')
    def test_terminal_metrics_use_epoch_not_reset_attempt(self, factory):
        trainer, callback = self.make_writer()
        for epoch in (trainer.metrics.epoch_train_metrics, trainer.metrics.epoch_valid_metrics):
            epoch.total_accuracy, epoch.total_loss, epoch.rankic = .2, .3, .4
        callback.add_total_metric()
        trainer.metrics.attempt_metrics.latest.assert_not_called()
        self.assertEqual(factory.return_value.add_scalars.call_args_list[0].args[1],
                         {'train': .2, 'valid': .2})


class PhaseScheduleTest(TestCase):
    def test_decayed_peak_warmup_continuation_and_state(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=.001)
        old = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=.001,
            step_size_up=12, cycle_momentum=False, mode='triangular2')
        for _ in range(30):
            optimizer.step()
            old.step()
        captured = capture_cycle(SimpleNamespace(optimizer=optimizer, scheduler=old))
        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertLess(captured['peaks'][0], .001)
        scheduler = PhaseCyclicLR(optimizer, **captured)
        values = [optimizer.param_groups[0]['lr']]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]['lr'])
        self.assertTrue(all(a <= b for a, b in zip(values[:9], values[1:10])))
        self.assertAlmostEqual(values[9], captured['peaks'][0])
        self.assertLess(values[10], values[9])
        state = scheduler.state_dict()
        clone = PhaseCyclicLR(optimizer, **captured)
        clone.load_state_dict(state)
        scheduler.step()
        clone.step()
        self.assertEqual(scheduler.get_last_lr(), clone.get_last_lr())


class CappedTopKTest(TestCase):
    def test_weights_cap_normalization_and_gradient(self):
        loss = CappedSoftTopKLoss(.25)
        pred = torch.tensor([8., 3., 2., 1., 0., -.5, -1., -2.], dtype=torch.double, requires_grad=True)
        weights = loss.weights(pred)
        self.assertAlmostEqual(weights.sum().item(), 1.)
        self.assertLessEqual(weights.max().item(), .5 + 1e-10)
        self.assertTrue(torch.autograd.gradcheck(loss.weights, (pred,)))
        label = torch.arange(8, dtype=torch.double)
        loss.forward(pred, label).backward()
        assert pred.grad is not None
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_uniform_full_selection_and_singleton(self):
        for n in (1, 10):
            pred = torch.randn(n, dtype=torch.double, requires_grad=True)
            loss = CappedSoftTopKLoss(1.)
            weights = loss.weights(pred)
            torch.testing.assert_close(weights, torch.full_like(pred, 1 / n))
            (weights * torch.randn(n)).sum().backward()
            torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_multi_output_equal_and_extreme_scores(self):
        loss = CappedSoftTopKLoss(.2)
        for pred in (torch.zeros(20, 2), torch.full((20, 2), 1e10), torch.randn(20, 2) * 1e8):
            w = loss.weights(pred)
            torch.testing.assert_close(w.sum(0), torch.ones(2))
            self.assertTrue(torch.isfinite(w).all())
            self.assertLessEqual(w.max().item(), .25 + 1e-6)

class BaselineIsolationTest(TestCase):
    def test_baseline_preserves_rng_modes_and_history_on_success_and_failure(self):
        import random
        import numpy as np
        from src.res.model.util.trainer.phase_transition import validate_recalled_model

        class Metrics:
            def __init__(self):
                self.stage_metrics: dict[str, Any] = {'history': ['keep']}
                self.aggregator = {}
                self.accuracy_function = SimpleNamespace()

            @property
            def epoch_valid_metrics(self):
                if 'valid' not in self.stage_metrics:
                    self.stage_metrics['valid'] = SimpleNamespace(new=lambda *a: None,
                        append=lambda *a: None, accuracies_dict={'global.ic': .2}, losses_dict={'global.ccc': .4})
                return self.stage_metrics['valid']

            def calculate(self, *args):
                self.batch_metrics = SimpleNamespace(total_loss=torch.tensor(.4), close=lambda: None)

        net = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(.5))
        net.train()
        net[1].eval()  # Preserve mixed submodule modes as well as the top-level mode.
        metrics = Metrics()
        def forward(batch):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(net.training)
            torch.rand(2)
            random.random()
            np.random.rand()
            return SimpleNamespace(output=SimpleNamespace(empty=False))
        trainer = SimpleNamespace(metrics=metrics, model=SimpleNamespace(net=net, get_batch_data=forward),
            data=SimpleNamespace(val_dataloader=lambda: [1, 2]), batch_warm_up=0, batch_aftermath=100)
        rng = torch.get_rng_state().clone()
        py_rng, np_rng = random.getstate(), np.random.get_state()
        self.assertEqual(validate_recalled_model(trainer), ({'global.ic': .2}, {'global.ccc': .4}))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(random.getstate(), py_rng)
        np.testing.assert_equal(np.random.get_state(), np_rng)
        self.assertEqual(metrics.stage_metrics, {'history': ['keep']})
        self.assertTrue(net.training)
        self.assertFalse(net[1].training)
        trainer.model.get_batch_data = Mock(side_effect=RuntimeError('injected'))
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            validate_recalled_model(trainer)
        self.assertTrue(net.training)
        torch.testing.assert_close(torch.get_rng_state(), rng)


class CurrentEpochSelectionTest(TestCase):
    def test_assessment_occurs_after_current_validation_is_collected(self):
        from src.res.model.util.metric.trainer_metrics import TrainerMetrics
        metrics = Mock(spec=TrainerMetrics)
        latest = []
        metrics.collect_fit_epoch.side_effect = lambda: latest.append('current')
        best = Mock()
        best.assess.side_effect = lambda status, given: self.assertEqual(latest, ['current'])
        metrics.model.submodels = {'best': best}
        TrainerMetrics.on_validation_epoch_end(metrics)
        best.assess.assert_called_once_with(metrics.status, metrics)
