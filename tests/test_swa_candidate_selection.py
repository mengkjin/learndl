"""Tests for independent attempt selection and anchored SWA continuation."""
import math
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from src.proj.util.functional.device import Device
from src.res.algo.nn.model.resnet_gru import resnet_gru_simp
from src.res.model.model_module.module.nn import NNPredictor
from src.res.model.model_module.util.swa import EnsembleSWABest, EnsembleSWALast, SWAModel, update_swa_bn
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import BatchData, BatchInput, BatchOutput, ModelDict
from src.res.model.util.metric.aggregator import MetricAggregator
from src.res.model.util.metric.trainer_metrics import TrainerMetrics
from src.res.model.util.storage.checkpoint import Checkpoint
from src.res.model.util.storage.deposition import Deposition
from src.res.model.util.trainer.base_trainer import BaseTrainer
from src.res.model.util.trainer.optimizer import Optimizer


class SWASelectionTests(TestCase):
    def test_select_checkpoints_by_observed_phase_and_fixed_accuracy(self):
        checkpoints = Mock()
        metrics = Mock()
        metrics.aggregator = MetricAggregator()
        metrics.model_metrics.accuracy_verdict = {'ic': 1.}
        metrics.compare_epochs.side_effect = lambda a, b: b is None or a.valid_accuracies['ic'] > b.valid_accuracies['ic']

        def point(epoch, phase, ic):
            return SimpleNamespace(epoch=epoch, phase=phase, valid_accuracies={'ic': ic})

        candidates = EnsembleSWABest(checkpoints, n_best=2)
        for epoch, ic in enumerate((.3, .1, .4, .3)):
            metrics.attempt_metrics.latest_epoch.return_value = point(epoch, 0, ic)
            candidates.assess(None, metrics)
        self.assertEqual(candidates.source_epochs(), [(0, 0), (2, 0)])
        checkpoints.disjoin.assert_called_once_with(candidates, 1, 0)

        for count, expected in [(1, [(14, 1)]), (2, [(14, 1), (15, 1)]),
                                (5, [(12, 1), (14, 1), (15, 1), (18, 1)])]:
            swa = EnsembleSWALast(checkpoints, n_last=count)
            for phase, epochs, best in [(0, range(10), 6), (1, range(11, 19), 14)]:
                for epoch in epochs:
                    metrics.attempt_metrics.latest_epoch.return_value = point(epoch, phase, -abs(epoch - best))
                    swa.assess(None, metrics)
            self.assertEqual(swa.source_epochs(), expected)
            self.assertTrue(all(phase == 1 for _, phase in swa.retained))

    def test_bn_restores_mixed_modes_and_momentum_on_error(self):
        net = nn.Sequential(nn.BatchNorm1d(2, momentum=.2), nn.Dropout()).eval()
        net[1].train()

        def batches():
            yield torch.randn(6, 2), {}
            raise RuntimeError('injected')

        with self.assertRaisesRegex(RuntimeError, 'injected'):
            update_swa_bn(batches(), net)
        self.assertFalse(net.training)
        self.assertFalse(net[0].training)
        self.assertTrue(net[1].training)
        self.assertEqual(net[0].momentum, .2)

    def test_bn_prepares_current_batch_without_training_hooks(self):
        batch = BatchInput.random(4, 3, 2, 1)
        trainer = Mock()
        trainer.callback.callbacks = [SimpleNamespace(
            auxiliary_batch_kwargs=lambda batch, training: {'scale': batch.y})]
        trainer.data.train_dataloader.return_value = [batch]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(1)

            def forward(self, x, scale):
                return self.bn(scale)

        swa = SWAModel(Net())
        trainer.device = torch.device('cpu')
        averaged = swa.update_bn(trainer).module
        bn = averaged.get_submodule('bn')
        assert isinstance(bn, nn.BatchNorm1d) and bn.running_mean is not None
        torch.testing.assert_close(bn.running_mean, batch.y.mean(dim=0))
        self.assertEqual(batch.kwargs, {})
        trainer.on_train_batch_start.assert_not_called()

    def test_candidate_pipeline_with_two_output_heads(self):
        for variant in ('learned', 'pearson'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                self.run_candidate_pipeline(variant, Path(directory))

    def run_candidate_pipeline(self, variant, directory):
        torch.manual_seed(27)
        name = f'resnet_gru_simp_15m_{variant}_swa'
        cfg = ModelConfig(None, schedule_name=name, resume=0, vb_level='never',
                          override={'model.submodels': ['best', 'swalast', 'swa'], 'train.swa.lr_ratio': .1})
        self.assertEqual(cfg.swa_settings, (5, .1))
        cfg.device = Device(try_cuda=False, policy='cpu_only')
        trainer: Any = object.__new__(BaseTrainer)
        trainer._config = cfg
        trainer._status = SimpleNamespace(stage='fit', epoch=0, phase=0, model_date=20230101,
                                          model_num=0, model_submodel='best', next_attempt=0,
                                          next_redo=0, attempt_key='Trial0-0')
        trainer._callback = SimpleNamespace(callbacks=[])
        trainer._checkpoint = Checkpoint()
        trainer._checkpoint.model_key = 'swa_test'
        trainer._model = model = NNPredictor().bound_with(trainer)
        trainer._metrics = metrics = TrainerMetrics(trainer)
        trainer.on_before_stack_model = model.on_before_stack_model
        trainer.on_before_dump_model = Mock()
        trainer.write_status_message = Mock()
        trainer.on_swa_continuation_end = Mock()

        def batches():
            return [BatchInput(torch.randn(8, 3, 16, 6), torch.randn(8, 1), None,
                               torch.zeros(8, 2, dtype=torch.long), torch.ones(8, 1, dtype=torch.bool),
                               np.array([1]), np.arange(8)) for _ in range(2)]

        train, valid = batches(), batches()
        trainer._data = SimpleNamespace(train_dataloader=lambda: train, val_dataloader=lambda: valid)
        deposition = Deposition(cfg.base_path)
        deposition.model_path = lambda model_num, model_date, submodel='best': directory / submodel
        trainer._deposition = deposition
        net = resnet_gru_simp(6, inday_dim=16, hidden_mean_pool=variant == 'pearson')
        model.net = model.wrap_net(net)
        model.reset_submodels()
        model.optimizer = Optimizer(net, cfg, lr_multiplier=.3)
        metrics.new_model(model, {})
        with patch('src.res.model.util.storage.checkpoint.PATH', SimpleNamespace(checkpoint=directory)), \
                patch('src.res.model.util.metric.stage_metrics.Save.dfs') as save:
            # Produce and assess a real checkpoint using the regular metric path.
            for epoch in range(2):
                trainer.status.epoch = epoch
                metrics.new_fit_epoch()
                datasets: tuple[tuple[Literal['train', 'valid'], list[BatchInput]], ...] = (
                    ('train', train), ('valid', valid))
                for dataset, values in datasets:
                    net.train(dataset == 'train')
                    accumulator = metrics.epoch_train_metrics if dataset == 'train' else metrics.epoch_valid_metrics
                    with torch.set_grad_enabled(dataset == 'train'):
                        for index, batch in enumerate(values):
                            metrics.calculate(dataset, index, BatchData(batch, BatchOutput(model.forward(batch))))
                            _ = metrics.batch_metrics.total_loss
                            accumulator.append(metrics.batch_metrics)
                            if dataset == 'train':
                                model.optimizer.backward(metrics.batch_metrics)
                            metrics.batch_metrics.close()
                model.on_validation_epoch_end()
                metrics.on_validation_epoch_end()

            method = model.submodels['swa']
            source = method.source_epochs()[0]
            anchor = trainer.checkpoint.load(*source)
            self.assertTrue(anchor['optimizer']['state'])
            expected = deepcopy(model.persist_net())
            expected.load_state_dict(anchor['net'])
            reference = Optimizer(expected, cfg)
            reference.optimizer.load_state_dict(deepcopy(anchor['optimizer']))
            for group in reference.optimizer.param_groups:
                group['lr'] = .001 * .3 * .1
            runtime = deepcopy(net.state_dict())
            rng = torch.get_rng_state().clone()
            ref_states = []
            with torch.random.fork_rng():
                for _ in range(5):
                    expected.train()
                    for batch in train:
                        data = BatchData(batch, BatchOutput(expected(batch.x)))
                        loss = method.loss_function.losses(data)
                        reference.optimizer.zero_grad()
                        torch.stack(list(loss.values())).sum().backward()
                        reference.clip_gradients()
                        reference.optimizer.step()
                    ref_states.append(deepcopy(expected.state_dict()))

            # Simulate a later loss-weight transition after the anchor was selected.
            metrics.aggregator.inject('loss', {'mse': 0., 'pearson': 0.})
            model.stack_model()
            model.dump_model()
            saved = torch.load(directory / 'swa/state_dict.pt')
            for key, value in expected.named_parameters():
                torch.testing.assert_close(saved[key], torch.stack([state[key] for state in ref_states]).mean(0))
            for key, value in net.state_dict().items():
                torch.testing.assert_close(value, runtime[key])
            torch.testing.assert_close(torch.get_rng_state(), rng)
            records = metrics.model_metrics.submodel_attempts
            self.assertAlmostEqual(records['Trial0-0', 'best']['score_change'], 0., places=6)
            self.assertTrue(math.isfinite(records['Trial0-0', 'swa']['score']))
            self.assertEqual(records['Trial0-0', 'swa']['loss_weights'], {})
            self.assertEqual(records['Trial0-0', 'swa']['swa_epochs'], 5)
            swa_record = records['Trial0-0', 'swa']
            trainer.on_swa_continuation_end.assert_called_once_with(swa_record)
            trace = swa_record['swa_trace']
            self.assertEqual([point['step'] for point in trace], list(range(6)))
            self.assertAlmostEqual(trace[0]['valid_rankic'], records['Trial0-0', 'best']['valid_rankic'], places=6)
            self.assertTrue(all(math.isfinite(point['train_loss']) for point in trace[1:]))
            self.assertTrue(all(math.isfinite(point['valid_rankic']) for point in trace))

            # Reversing collection order must not change the averaged weights or BN.
            repeated_swa = model.collect('swa').state_dict
            model.collect('swalast')
            model.collect('best')
            assert repeated_swa is not None
            for key, value in saved.items():
                torch.testing.assert_close(value, repeated_swa[key])
            torch.testing.assert_close(torch.get_rng_state(), rng)
            self.assertEqual(set(metrics.model_metrics.selected_submodel_attempts), {'best', 'swalast', 'swa'})
            self.assertEqual(len(save.call_args.args[0]['submodel_attempts']), 3)

            # Stack another attempt with distinct weights. Only SWA should choose it.
            records['Trial1-0', 'best'] = {**records['Trial0-0', 'best'], 'attempt': 'Trial1-0'}
            records['Trial1-0', 'swa'] = {**records['Trial0-0', 'swa'], 'attempt': 'Trial1-0', 'score': 1.}
            for submodel in cfg.submodels:
                state = torch.load(directory / submodel / 'state_dict.pt')
                deposition.stack_model(ModelDict(state_dict=state), 'Trial0-0', 0, 20230101, submodel)
                if submodel in ('best', 'swa'):
                    state[next(iter(state))].add_(1)
                    deposition.stack_model(ModelDict(state_dict=state), 'Trial1-0', 0, 20230101, submodel)
            selected = metrics.model_metrics.select_submodel_attempts(cfg.submodels)
            self.assertEqual(selected, {'best': 'Trial0-0', 'swalast': 'Trial0-0', 'swa': 'Trial1-0'})
            for submodel, attempt in selected.items():
                target = directory / submodel / attempt / 'state_dict.pt'
                expected_state = torch.load(target)
                deposition.dump_stacked_model(attempt, 0, 20230101, submodel)
                dumped_state = torch.load(directory / submodel / 'state_dict.pt')
                for key, value in expected_state.items():
                    torch.testing.assert_close(value, dumped_state[key])
            for record in trainer.checkpoint.epoch_maps.values():
                record.future.result()
