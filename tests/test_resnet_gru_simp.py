"""Identity residual, boundary semantics and real training configuration."""
import unittest
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from src.res.algo.nn.api import get_nn_module
from src.res.algo.nn.model.resnet_gru import resnet_gru_simp, _IdentityIntradayResidual
from src.res.algo.nn.loss.loss import Loss, MSE, HiddenCorr
from src.res.model.util.config import ModelConfig
from src.res.model.callback.consolidate import ConsolidateCallBack
from src.res.model.util.core import BatchInput, BatchOutput, BatchData
from src.res.model.util.metric.functions import LossFunction


class SimpTests(unittest.TestCase):
    def test_shapes_projection_and_gradients(self):
        self.assertIs(get_nn_module('resnet_gru_simp'), resnet_gru_simp)
        for bars in (8, 16):
            for project in (False, None, True, 'linear'):
                for features in (3, 6):
                    model = resnet_gru_simp(features, inday_dim=bars, enc_in=project,
                                           enc_in_dim=7, inday_resnet_dim=11, hidden_dim=9)
                    x = torch.randn(3, 4, features, bars).transpose(-1, -2).requires_grad_()
                    y, other = model(x)
                    self.assertEqual(y.shape, (3, 1))
                    self.assertEqual(other['hidden'].shape, (3, 9))
                    (y.square().mean() + other['hidden'].square().mean()).backward()
                    assert x.grad is not None
                    self.assertTrue(torch.isfinite(x.grad).all())
                    for p in model.parameters():
                        assert p.grad is not None
                        self.assertTrue(torch.isfinite(p.grad).all())

    def test_identity_and_gradient(self):
        block = _IdentityIntradayResidual(6)
        self.assertIsInstance(block.shortcut, nn.Identity)
        self.assertEqual(sum(p.numel() for p in block.shortcut.parameters()), 0)
        self.assertEqual(sum(isinstance(m, nn.Conv1d) for m in block.modules()), 2)
        self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in block.modules()))
        for p in block.branch.parameters():
            nn.init.zeros_(p)
        x = torch.randn(2, 6, 8, requires_grad=True)
        torch.testing.assert_close(block(x), x)
        block(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_replicate_and_independent_sequences(self):
        block = _IdentityIntradayResidual(1, act_type='relu')
        first, last = block.branch[0], block.branch[2]
        assert isinstance(first, nn.Conv1d) and isinstance(last, nn.Conv1d)
        assert first.bias is not None and last.bias is not None
        with torch.no_grad():
            first.weight.fill_(1)
            first.bias.zero_()
            last.weight.zero_()
            last.weight[:, :, 1] = 1
            last.bias.zero_()
        x = torch.tensor([[[1., 2., 4.]], [[10., 20., 40.]]])
        # Replicated 3-tap sums are [4,7,10], then add original values.
        torch.testing.assert_close(block(x), torch.tensor([[[5., 9., 14.]], [[50., 90., 140.]]]))
        torch.testing.assert_close(block(x)[:1], block(x[:1]))

    def test_projection_shared_and_flatten_after_add(self):
        model = resnet_gru_simp(3, inday_dim=8, enc_in=True, enc_in_dim=5).eval()
        x = torch.randn(2, 3, 8, 3)
        for p in model.inday_resnet.branch.parameters():
            nn.init.zeros_(p)
        projected = model.fc_enc_in(x)
        expected = model.daily_projection(projected.reshape(6, 8, 5).transpose(1, 2).flatten(1))
        actual = model.encode_intraday(x)
        torch.testing.assert_close(actual, expected.reshape(2, 3, 128))
        for b in range(2):
            for d in range(3):
                torch.testing.assert_close(actual[b:b+1, d:d+1], model.encode_intraday(x[b:b+1, d:d+1]))
        h = model.fc_rnn(actual)[:, -1]
        factors, pred = model.output(h)
        torch.testing.assert_close(pred, model.output.fc_map_out[-1](factors.mean(-1, keepdim=True)))

    def test_gru_dropout_and_parameter_count(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = resnet_gru_simp(6, inday_dim=16)
        self.assertFalse(any('dropout' in str(w.message) for w in caught))
        self.assertEqual(model.fc_rnn.gru.dropout, 0)
        self.assertEqual(sum(p.numel() for p in model.fc_rnn.parameters()), 37248)
        self.assertEqual(resnet_gru_simp(6, inday_dim=16, rnn_layers=2).fc_rnn.gru.dropout, .1)

    def test_invalid_inputs(self):
        invalid: list[dict[str, Any]] = [dict(resnet_kernel_size=k) for k in (0, 2, True, 1.5)]
        invalid += [dict(rnn_layers=4), dict(enc_in='bad'), dict(hidden_dim=0), dict(rnn_type='lstm')]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                resnet_gru_simp(6, inday_dim=8, **kwargs)
        model = resnet_gru_simp(6, inday_dim=8)
        for shape in ((2, 8, 6), (2, 3, 6, 8), (2, 0, 8, 6)):
            with self.assertRaises(ValueError):
                model(torch.randn(shape))
        with self.assertRaises(TypeError):
            model(torch.ones(2, 3, 8, 6, dtype=torch.long))

    def test_real_config_and_training_metric(self):
        config = ModelConfig(None, schedule_name='resnet_gru_simp_15m', resume=0, vb_level='never')
        self.assertEqual(config.model_module, 'resnet_gru_simp')
        self.assertEqual(config.criterion_loss, {'mse_hcorr': {'alpha': .01}})
        self.assertEqual(config.label_normalization, 'standardize')
        self.assertTrue(config.input_data_prenorm['15m']['histnorm'])
        self.assertEqual(config.weight_scheme('fit'), 'equal')
        callbacks = ConsolidateCallBack(config)
        self.assertNotIn('SpecificCB_Global2Top', [type(cb).__name__ for cb in callbacks.callbacks])
        self.assertEqual(config.criterion_loss, {'mse_hcorr': {'alpha': .01}})
        params = dict(config.model_param[0])
        model = get_nn_module(config.model_module)(input_dim=6, inday_dim=16, **params)
        self.assertEqual(model.fc_rnn.gru.num_layers, 1)
        self.assertIsInstance(model.fc_enc_in, nn.Identity)
        x, labels = torch.randn(4, 30, 16, 6), torch.randn(4, 1)
        pred, other = model(x)
        inp = BatchInput(x, labels, torch.arange(1., 5.).unsqueeze(-1),
                         torch.zeros(4, 2, dtype=torch.long), torch.ones(4, 1, dtype=torch.bool),
                         np.array([1]), np.arange(4))
        data = BatchData(inp, BatchOutput((pred, other)))
        values = LossFunction(config.criterion_loss, model).losses(data)
        self.assertEqual(set(values), {'mse', 'hidden_corr'})
        torch.stack(list(values.values())).sum().backward()


class MSEHiddenTests(unittest.TestCase):
    def test_parts_weights_dim_and_gradients(self):
        for dim in (None, 0):
            for alpha in (0., .01):
                pred = torch.randn(9, 1, requires_grad=True)
                label = torch.randn(9, 1)
                hidden = torch.randn(9, 4, requires_grad=True)
                weight = torch.arange(1., 10.).unsqueeze(-1)
                result = Loss.get('mse_hcorr', alpha=alpha)(pred, label, hidden, weight, dim)
                assert isinstance(result, dict)
                torch.testing.assert_close(result['mse'], MSE()(pred, label, weight, dim))
                expected_corr = HiddenCorr()(hidden=hidden)
                assert isinstance(expected_corr, torch.Tensor)
                torch.testing.assert_close(result['hidden_corr'], alpha * expected_corr)
                total = torch.stack([v.sum() for v in result.values()]).sum()
                total.backward()
                for t in (pred, hidden):
                    assert t.grad is not None
                    self.assertTrue(torch.isfinite(t.grad).all())
                if alpha == 0:
                    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
