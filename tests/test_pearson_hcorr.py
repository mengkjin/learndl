"""Pearson composite loss and the two simp single-factor schedules."""
import unittest

import torch
from torch import nn

from src.res.algo.nn.loss.loss import Loss, PearsonLoss, HiddenCorr
from src.res.algo.nn.api import get_nn_module
from src.res.model.util.config import ModelConfig
from src.res.model.callback.consolidate import ConsolidateCallBack
from src.res.model.util.core import BatchInput, BatchOutput, BatchData
from src.res.model.util.metric.functions import LossFunction
import numpy as np


class PearsonHiddenTests(unittest.TestCase):
    def test_parts_and_gradients(self):
        for dim in (None, 0):
            for weighted in (False, True):
                for alpha in (0., .01):
                    pred = torch.randn(9, 1, requires_grad=True)
                    label = torch.randn(9, 1)
                    hidden = torch.randn(9, 4, requires_grad=True)
                    weight = torch.arange(1., 10.).unsqueeze(-1) if weighted else None
                    parts = Loss.get('pearson_hcorr', alpha=alpha)(pred, label, hidden, weight, dim)
                    assert isinstance(parts, dict)
                    self.assertEqual(set(parts), {'pearson', 'hidden_corr'})
                    torch.testing.assert_close(parts['pearson'], PearsonLoss()(pred, label, weight, dim))
                    corr = HiddenCorr()(hidden=hidden)
                    assert isinstance(corr, torch.Tensor)
                    torch.testing.assert_close(parts['hidden_corr'], alpha * corr)
                    torch.stack([v.sum() for v in parts.values()]).sum().backward()
                    for t in (pred, hidden):
                        assert t.grad is not None
                        self.assertTrue(torch.isfinite(t.grad).all())
                    if alpha == 0:
                        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))

    def test_schedules_and_metric_integration(self):
        baseline = ModelConfig(None, schedule_name='resnet_gru_simp_15m',
                               resume=0, short_test=False, vb_level='never')
        for suffix in ('learned', 'pearson'):
            config = ModelConfig(None, schedule_name=f'resnet_gru_simp_15m_{suffix}',
                                 resume=0, short_test=False, vb_level='never')
            expected = dict(baseline.model_param[0])
            if suffix == 'learned':
                expected['hidden_mean_pool'] = False
            self.assertEqual(config.model_param, [expected])
            key = 'pearson_hcorr' if suffix == 'pearson' else 'mse_hcorr'
            self.assertEqual(config.criterion_loss, {key: {'alpha': .01}})
            self.assertEqual(config.input_data_prenorm, baseline.input_data_prenorm)
            self.assertEqual(config.weight_scheme('fit'), 'equal')
            self.assertEqual(config.weight_scheme('test'), 'equal')
            callbacks = ConsolidateCallBack(config)
            self.assertNotIn('SpecificCB_Global2Top', [type(cb).__name__ for cb in callbacks.callbacks])
            model = get_nn_module(config.model_module)(input_dim=6, inday_dim=16, **expected)
            if suffix == 'learned':
                self.assertTrue(any(isinstance(m, nn.Linear) and m.out_features == 1
                                    for m in model.modules()))
            x, labels = torch.randn(4, 30, 16, 6), torch.randn(4, 1)
            inp = BatchInput(x, labels, torch.ones(4, 1), torch.zeros(4, 2, dtype=torch.long),
                             torch.ones(4, 1, dtype=torch.bool), np.array([1]), np.arange(4))
            data = BatchData(inp, BatchOutput(model(x)))
            parts = LossFunction(config.criterion_loss, model).losses(data)
            self.assertEqual(set(parts), {'pearson' if suffix == 'pearson' else 'mse', 'hidden_corr'})
            torch.stack(list(parts.values())).sum().backward()
