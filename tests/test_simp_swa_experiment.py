"""Paired best/SWALast experiment: configuration and real-network collection."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np
import torch
import yaml

from src.res.algo.nn.model.resnet_gru import resnet_gru_simp
from src.res.model.model_module.util.swa import EnsembleBestOne, EnsembleSWALast
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import BatchInput

ROOT = Path(__file__).resolve().parents[1]


class PairedSwaTests(unittest.TestCase):
    def test_effective_configs_only_change_identity_and_submodels(self):
        for variant in ('learned', 'pearson'):
            name = f'resnet_gru_simp_15m_{variant}'
            folder = ROOT / 'configs/schedule/current'
            original = yaml.safe_load((folder / f'{name}.yaml').read_text())
            paired = yaml.safe_load((folder / f'{name}_swa.yaml').read_text())
            self.assertEqual(paired['model'].pop('name'), name + '_swa')
            self.assertEqual(paired['model'].pop('submodels'), ['best', 'swalast'])
            original['model'].pop('name')
            original['model'].pop('submodels')
            self.assertEqual(paired, original)
            config = ModelConfig(None, schedule_name=name + '_swa', resume=0, vb_level='never')
            self.assertEqual(config.submodels, ['best', 'swalast'])
            self.assertEqual(config.model_module, 'resnet_gru_simp')

    def test_collection_averages_weights_and_recomputes_bn_for_both_heads(self):
        torch.manual_seed(7)
        for mean_pool in (False, True):
            net = resnet_gru_simp(6, inday_dim=16, hidden_mean_pool=mean_pool).eval()
            checkpoints = {}
            ckpt = Mock()
            ckpt.load.side_effect = lambda ep, ph: checkpoints[(ep, ph)]
            best, swa = EnsembleBestOne(ckpt), EnsembleSWALast(ckpt)
            metrics = Mock()
            metrics.compare_epochs.side_effect = lambda a, b: b is None or a.score > b.score
            # Best epoch 8: SWALast should select 3, 6, 8, 9, 12, not the last five.
            for epoch in range(19):
                with torch.no_grad():
                    next(net.parameters()).add_(0.001)
                checkpoints[(epoch, 0)] = {'net': deepcopy(net.state_dict())}
                metrics.attempt_metrics.latest_epoch.return_value = SimpleNamespace(
                    epoch=epoch, phase=0, score=-abs(epoch - 8))
                best.assess(Mock(), metrics)
                swa.assess(Mock(), metrics)
            chosen = [(ep, 0) for ep in (3, 6, 8, 9, 12)]
            self.assertEqual(swa.adjacent_epochs[:5], chosen)
            batches = [BatchInput(torch.randn(6, 30, 16, 6), torch.randn(6, 1), None,
                                  torch.zeros(6, 2, dtype=torch.long), torch.ones(6, 1, dtype=torch.bool),
                                  np.array([1]), np.arange(6)) for _ in range(2)]
            trainer = Mock()
            trainer.device = torch.device('cpu')
            trainer.model.persist_net.return_value = net
            trainer.data.train_dataloader.return_value = batches
            untouched = deepcopy(net.state_dict())
            selected = best.collect(trainer)
            averaged = swa.collect(trainer)
            for key, value in selected.state_dict().items():
                torch.testing.assert_close(value, checkpoints[(8, 0)]['net'][key])
            for key, value in averaged.named_parameters():
                expected = torch.stack([checkpoints[ep]['net'][key] for ep in chosen]).mean(0)
                torch.testing.assert_close(value, expected)
            for layer in averaged.modules():
                if isinstance(layer, torch.nn.BatchNorm1d):
                    assert layer.num_batches_tracked is not None and layer.running_var is not None
                    self.assertEqual(layer.num_batches_tracked.item(), len(batches))
                    self.assertTrue(torch.isfinite(layer.running_var).all())
            for key, value in net.state_dict().items():
                torch.testing.assert_close(value, untouched[key])
            output, _ = averaged.eval()(batches[0].x)
            self.assertEqual(output.shape, (6, 1))
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
