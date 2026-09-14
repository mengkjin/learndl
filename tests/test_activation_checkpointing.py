"""CPU regression tests; optional CUDA memory smoke test.

Run: python -m unittest discover -s tests -p test_activation_checkpointing.py
On a CUDA server, set AC_TEST_CUDA_BATCH_SIZE to the failing batch's stock count
to compare complete forward/backward allocation peaks (synthetic FP32 inputs).
"""
import copy
import gc
import os
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import Mock, patch
import weakref

import torch
from torch import nn
import yaml

from src.res.algo.nn.layer.checkpoint import ActivationCheckpointMixin, is_cuda_oom
from src.res.algo.nn.model.transformer_gru import transformer_gru
from src.res.model.callback.consolidate import ConsolidateCallBack
from src.res.model.callback.fit.activation_checkpointing import ActivationCheckpointing
from src.res.model.callback.fit.early_stop import EarlyStoppage
from src.res.model.callback.fit.retrain import BadAttemptRetrain
from src.res.model.callback.specific.global2top import SpecificCB_Global2Top
from src.res.model.callback.monitor.summary import SummaryWriter
from src.res.model.util.advance.torch_compile import TorchCompiler
from src.res.model.util.storage.deposition import Deposition
from src.res.model.util.trainer.base_trainer import BaseTrainer
from src.res.model.util.trainer.future_utils import FutureUtils
from src.res.model.util.trainer.status import TrainerStatus
from src.res.model.util.config import ModelConfig
from src.res.model.util.trainer.base_callback import BaseCallBack
from src.res.model.util.trainer.predictor_model import PredictorModel
from src.res.model.util.storage.checkpoint import Checkpoint
from src.res.model.util.core import BatchInput, BatchOutput

ROOT = Path(__file__).resolve().parents[1]


class TinyNet(ActivationCheckpointMixin, nn.Module):
    activation_checkpoint_regions = ('encode',)

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def encode(self, x):
        return self.linear(x).tanh()

    def forward(self, x):
        return self.checkpoint_region('encode', x)


def make_trainer(enabled=True):
    # Bypass the production singleton while retaining real trainer/callback APIs.
    trainer = object.__new__(BaseTrainer)
    BaseTrainer.__init__(trainer)
    trainer._config = Mock(spec=ModelConfig,
        callback_kwargs={'ActivationCheckpointing': {'enabled': enabled}},
        module_type='nn', submodels=['best'],
    )
    trainer._status = TrainerStatus()
    trainer.status.stage = 'fit'
    trainer.status.model_date = 20231201
    trainer.status.model_num = 0
    trainer.status.first_iteration_printed = True
    trainer._metrics = Mock()
    trainer._model = Mock(spec=PredictorModel, persist_net=lambda: TinyNet())
    trainer.batch_input = Mock(spec=BatchInput, x=torch.ones(3, 2))
    trainer.batch_idx = 0
    callback = ActivationCheckpointing(trainer, enabled=enabled)
    manager = object.__new__(ConsolidateCallBack)
    callbacks: list[BaseCallBack] = [callback] if callback else []
    manager.callbacks = callbacks
    trainer._callback = manager
    return trainer, callback


class RegionTests(unittest.TestCase):
    def test_transformer_outputs_gradients_and_state_dict(self):
        torch.manual_seed(5)
        plain = transformer_gru(6, enc_in_dim=32, dropout=.1)
        checked = copy.deepcopy(plain)
        checked.set_activation_checkpointing(True)
        checked.load_state_dict(plain.state_dict(), strict=True)
        self.assertEqual(tuple(plain.state_dict()), tuple(checked.state_dict()))
        self.assertEqual(sum(p.numel() for p in plain.parameters()), 73698)
        x = torch.randn(4, 30, 16, 6)  # ordinary data has no requires_grad
        weights = torch.randn(4, 64)
        results = []
        for model in (plain, checked):
            torch.manual_seed(10)
            y, other = model(x)
            (y.square().mean() + (other['hidden'] * weights).mean()).backward()
            results.append(y.detach())
        torch.testing.assert_close(*results, atol=0, rtol=0)
        for a, b in zip(plain.parameters(), checked.parameters()):
            self.assertIsNotNone(b.grad)
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
        for a, b in zip(plain.buffers(), checked.buffers()):
            torch.testing.assert_close(a, b)

    def test_eval_no_grad_and_disabled_bypass(self):
        model = TinyNet()
        x = torch.randn(3, 2)
        with patch('src.res.algo.nn.layer.checkpoint.checkpoint', side_effect=AssertionError('called')):
            model(x)
            model.set_activation_checkpointing(True)
            model.eval()(x)
            model.train()
            with torch.no_grad():
                model(x)

    def test_invalid_regions(self):
        for regions in ((), 'encode', ('absent',), ('linear_weight',), (42,)):
            model = TinyNet()
            # Deliberately inject malformed runtime configuration.
            setattr(model, "activation_checkpoint_regions", regions)
            with self.assertRaises(ValueError):
                model.set_activation_checkpointing(True)

    def test_region_preserves_nested_results_and_gradients(self):
        class StructuredNet(ActivationCheckpointMixin, nn.Module):
            activation_checkpoint_regions = ('encode',)

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def encode(self, x):
                value = self.linear(x).tanh()
                return value, {'squared': value.square()}

        model = StructuredNet()
        reference = copy.deepcopy(model)
        model.set_activation_checkpointing(True)
        x = torch.randn(3, 2)
        actual = model.checkpoint_region('encode', x)
        expected = reference.checkpoint_region('encode', x)
        torch.testing.assert_close(actual, expected)
        for value, other in (actual, expected):
            (value.sum() + other['squared'].sum()).backward()
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_saved_encoder_storage_reduced(self):
        model = transformer_gru(6, enc_in_dim=32)
        x = torch.randn(4, 30, 16, 6)
        sizes = []
        for enabled in (False, True):
            model.set_activation_checkpointing(enabled)
            excluded = {p.untyped_storage().data_ptr() for p in model.parameters()}
            excluded.add(x.untyped_storage().data_ptr())
            storage = {}
            def pack(t):
                s = t.untyped_storage()
                if s.data_ptr() not in excluded:
                    storage[s.data_ptr()] = s.nbytes()
                return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                output = model(x)
            sizes.append(sum(storage.values()))
            del output
        self.assertLess(sizes[1], sizes[0] / 5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; server acceptance still required')
    def test_cuda_checkpoint_forward_backward(self):
        batch = int(os.environ.get('AC_TEST_CUDA_BATCH_SIZE', '64'))
        def measure(enabled):
            torch.cuda.reset_peak_memory_stats()
            model = transformer_gru(6, enc_in_dim=32).cuda()
            model.set_activation_checkpointing(enabled)
            y, other = model(torch.randn(batch, 30, 16, 6, device='cuda'))
            (y.square().mean() + other['hidden'].square().mean()).backward()
            torch.cuda.synchronize()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            return torch.cuda.max_memory_allocated()
        peaks = []
        for enabled in (False, True):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                peaks.append(measure(enabled))
            except torch.cuda.OutOfMemoryError:
                if enabled:
                    raise
                peaks.append(None)
        print(f'CUDA synthetic batch={batch}: normal/checkpoint peak allocated bytes={peaks}')
        if peaks[0] is not None:
            self.assertLess(peaks[1], peaks[0])


class CallbackTests(unittest.TestCase):
    def test_config_defaults_and_schedule(self):
        def read(path):
            return yaml.safe_load((ROOT / path).read_text())
        self.assertFalse(read('configs/model/callbacks.yaml')['ActivationCheckpointing']['enabled'])
        self.assertFalse(read('configs/model/default/optional.yaml')['callbacks.ActivationCheckpointing']['enabled'])
        self.assertIn('ActivationCheckpointing', read('configs/model/train.yaml')['callbacks'])
        self.assertIn('ActivationCheckpointing', read('configs/model/default/required.yaml')['train.callbacks'])
        self.assertTrue(read('configs/schedule/current/transformer_gru_15m.yaml')['callbacks']['ActivationCheckpointing']['enabled'])
        self.assertIs(ConsolidateCallBack.get_callback_class('ActivationCheckpointing'), ActivationCheckpointing)

    def test_explicit_opt_in_with_legacy_callback_list(self):
        trainer, _ = make_trainer()
        cast(Mock, trainer.config).configure_mock(
            callbackes=[], special={}, model_module='transformer_gru',
            model_clean_name='transformer_gru_15m',
        )
        manager = ConsolidateCallBack(trainer)
        self.assertEqual([type(cb).__name__ for cb in manager.callbacks], ['ActivationCheckpointing'])
        trainer.config.callback_kwargs['ActivationCheckpointing']['enabled'] = False
        self.assertFalse(ConsolidateCallBack(trainer).callbacks)

    def test_oom_before_first_batch_and_unsupported_module(self):
        trainer, callback = make_trainer()
        del trainer.batch_input
        self.assertTrue(callback.request_fit_restart(torch.cuda.OutOfMemoryError('CUDA OOM')))
        cast(Mock, trainer.config).configure_mock(module_type='boost')
        with self.assertRaises(ValueError):
            callback.on_fit_start_before()

    def test_cleanup_before_batch_attributes_exist(self):
        trainer, _ = make_trainer()
        del trainer.batch_input
        self.assertFalse(hasattr(trainer, 'dataloader'))
        trainer._checkpoint = Mock(spec=Checkpoint, epoch_maps={})
        trainer._deposition = Mock(spec=Deposition)
        trainer.status.total_models = 1
        with patch.object(FutureUtils, 'model', return_value=Mock(spec=PredictorModel)), \
                patch.object(FutureUtils, 'metrics', return_value=Mock()):
            BaseTrainer.on_fit_model_restart(trainer)
        self.assertFalse(hasattr(trainer, 'batch_input'))
        self.assertFalse(hasattr(trainer, 'dataloader'))
        self.assertEqual(trainer.status.total_models, 0)

    def test_disabled_unsupported_and_sticky(self):
        trainer, callback = make_trainer(False)
        self.assertFalse(trainer.callback.request_fit_restart(torch.cuda.OutOfMemoryError('CUDA OOM')))
        trainer, callback = make_trainer()
        trainer._model = Mock(spec=PredictorModel, persist_net=lambda: nn.Linear(2, 2))
        with self.assertRaises(ValueError):
            callback.on_new_attempt()
        for regions in ((), ('absent',)):
            net = TinyNet()
            net.activation_checkpoint_regions = regions
            trainer._model = Mock(spec=PredictorModel, persist_net=lambda: net)
            with self.assertRaises(ValueError):
                callback.on_new_attempt()
        net = TinyNet()
        trainer._model = Mock(spec=PredictorModel, persist_net=lambda: net)
        callback.on_new_attempt()
        self.assertFalse(net.activation_checkpointing_enabled)
        self.assertTrue(callback.request_fit_restart(torch.cuda.OutOfMemoryError('CUDA OOM')))
        callback.on_fit_model_restart()
        for date, num in ((20231201, 0), (20231201, 1), (20240601, 0)):
            trainer.status.model_date, trainer.status.model_num = date, num
            net = TinyNet()
            trainer._model = Mock(spec=PredictorModel, persist_net=lambda: net)
            callback.on_new_attempt()
            self.assertTrue(net.activation_checkpointing_enabled)
        self.assertFalse(callback.request_fit_restart(torch.cuda.OutOfMemoryError('CUDA OOM')))
        self.assertFalse(make_trainer()[1].active)

    def test_cpu_and_wrapped_errors(self):
        self.assertFalse(is_cuda_oom(RuntimeError('CPU out of memory')))
        wrapper = RuntimeError('compiler failed')
        wrapper.__cause__ = torch.cuda.OutOfMemoryError('CUDA OOM')
        self.assertTrue(is_cuda_oom(wrapper))
        trainer, callback = make_trainer()
        self.assertFalse(callback.request_fit_restart(RuntimeError('shape error')))

    def test_compiler_does_not_retry_oom(self):
        compiler = object.__new__(TorchCompiler)
        compiler._disabled = False
        compiler._raw = Mock()
        failure = RuntimeError('compiler wrapper')
        failure.__cause__ = torch.cuda.OutOfMemoryError('CUDA OOM')
        compiler._active = Mock(side_effect=failure)
        with self.assertRaisesRegex(RuntimeError, 'compiler wrapper'):
            compiler.run(torch.ones(1))
        compiler._raw.assert_not_called()

    def test_fit_restart_cleans_whole_model(self):
        for failure_stage in ('forward', 'backward', 'optimizer', 'valid'):
            with self.subTest(stage=failure_stage), tempfile.TemporaryDirectory() as tmp:
                self.run_restart_case(failure_stage, Path(tmp))

    def run_restart_case(self, failure_stage, path):
        trainer, callback = make_trainer()
        early = EarlyStoppage(trainer)
        quality = BadAttemptRetrain(trainer)
        phase = SpecificCB_Global2Top(trainer)
        summary = SummaryWriter(trainer)
        trainer.callback.callbacks += [early, quality, phase, summary]
        trainer._checkpoint = Mock(spec=Checkpoint, epoch_maps={}, clear_all=Mock())
        deposition = object.__new__(Deposition)
        deposition.model_path = lambda model_num, model_date, submodel='best': path / str(model_date) / str(model_num) / submodel
        trainer._deposition = deposition
        previous = path / 'completed.pt'
        previous.write_text('keep')
        staged = deposition.model_path(0, 20231201) / 'trial3'
        staged.mkdir(parents=True)
        (staged / 'weights.pt').write_text('discard')
        starts = []
        old_refs = []
        failures = []
        closed_writer = Mock()
        vars(summary)['writer'] = closed_writer

        def factory(_):
            torch.manual_seed(17)
            net = TinyNet()
            opt = torch.optim.Adam(net.parameters())
            predictor = Mock(spec=PredictorModel, net=net, persist_net=lambda: net, optimizer=opt)
            def fit():
                x = torch.randn(3, 2)
                trainer.batch_input = Mock(spec=BatchInput, x=x)
                output = net(x)
                trainer.batch_output = BatchOutput(output)
                if not failures:
                    failures.append(failure_stage)
                    # Simulate a later attempt/phase, not just epoch-zero OOM.
                    trainer.status.fitting_epochs.new_epoch()
                    trainer.status.current.attempt = 3
                    trainer.status.current.phase = 2
                    trainer.status.milestone_epochs.append(8)
                    phase.fitting_phase = 'top'
                    phase.loss_weights_set = True
                    quality.remain_nan_redo = 0
                    early.peak_epoch_metrics = 'old best'
                    if failure_stage != 'forward':
                        output.sum().backward()
                    if failure_stage in ('optimizer', 'valid'):
                        opt.step()
                    trainer.status.dataset = 'valid' if failure_stage == 'valid' else 'train'
                    old_refs.append(weakref.ref(net))
                    raise torch.cuda.OutOfMemoryError('CUDA injected OOM')
                self.assertEqual(trainer.status.attempt, 0)
                self.assertEqual(trainer.status.phase, 0)
                self.assertEqual(trainer.status.milestone_epochs, [])
                self.assertEqual(phase.fitting_phase, 'global')
                self.assertFalse(phase.loss_weights_set)
                self.assertIsNone(early.peak_epoch_metrics)
                self.assertEqual(quality.remain_nan_redo, quality.max_nan_redo)
                self.assertFalse(opt.state)
                self.assertTrue(net.activation_checkpointing_enabled)
                self.assertEqual(previous.read_text(), 'keep')
                self.assertFalse(staged.exists())
                gc.collect()
                self.assertIsNone(old_refs[0]())
                net(x).sum().backward()
                opt.step()
            predictor.fit = fit
            return predictor

        def start():
            trainer.status.on_fit_model_start()
            trainer._model = factory(trainer)
            callback.on_new_attempt()
            early.on_fit_model_start()
            quality.on_fit_model_start()
            phase.on_fit_model_start()
            starts.append((trainer.status.attempt, trainer.status.phase))

        def restart():
            BaseTrainer.on_fit_model_restart(trainer)
            for cb in trainer.callback.callbacks:
                cb.on_fit_model_restart()

        trainer.on_fit_model_start = start
        trainer.on_fit_model_restart = restart
        with patch.object(FutureUtils, 'model', side_effect=factory), patch.object(FutureUtils, 'metrics', return_value=Mock()):
            BaseTrainer.fit_current_model(trainer)
        self.assertEqual(starts, [(0, 0), (0, 0)])
        self.assertEqual(trainer.status.total_models, 1)
        self.assertTrue(callback.active)
        self.assertEqual(trainer.fit_model_restart_count, 1)
        closed_writer.close.assert_called_once()
        closed_writer.add_text.assert_called_once()

    def test_second_oom_and_non_oom_propagate(self):
        for failure in (torch.cuda.OutOfMemoryError('CUDA OOM'), ValueError('bad shape')):
            trainer, callback = make_trainer()
            def start():
                trainer._model = Mock(spec=PredictorModel, fit=Mock(side_effect=failure), persist_net=lambda: TinyNet())
                callback.on_new_attempt()
            trainer.on_fit_model_start = start
            trainer.on_fit_model_restart = Mock()
            with self.assertRaises(type(failure)):
                BaseTrainer.fit_current_model(trainer)
            self.assertEqual(trainer.on_fit_model_restart.call_count, int(is_cuda_oom(failure)))

    def test_setup_oom_is_not_retried(self):
        trainer, _ = make_trainer()
        trainer.on_fit_model_start = Mock(side_effect=torch.cuda.OutOfMemoryError('CUDA OOM'))
        trainer.on_fit_model_restart = Mock()
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            BaseTrainer.fit_current_model(trainer)
        trainer.on_fit_model_restart.assert_not_called()


if __name__ == '__main__':
    unittest.main()
