"""Daily hidden normalization and legacy GRU compatibility on synthetic CPU data."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from src.res.algo.nn import layer as Layer
from src.res.algo.nn.model import RNN
from src.res.model.util.config.config import AlgoConfig, ScheduleConfig


class LegacyDecoder(nn.Module):
    """Pre-change decoder layout, including its checkpoint key structure."""
    def __init__(self, hidden_dim, act_type, dec_mlp_layers, dec_mlp_dim,
                 dropout, hidden_as_factors, map_to_one=False, **kwargs):
        super().__init__()
        self.fc_dec_mlp = nn.Sequential()
        mlp_dim = dec_mlp_dim if dec_mlp_dim else hidden_dim
        for i in range(dec_mlp_layers):
            self.fc_dec_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim if i == 0 else mlp_dim, mlp_dim),
                Layer.Act.get_activation_fn(act_type), nn.Dropout(dropout)))
        out_dim = 1 if map_to_one else hidden_dim
        self.fc_hid_out = (nn.Sequential(nn.Linear(mlp_dim, out_dim), nn.BatchNorm1d(out_dim))
                           if hidden_as_factors else nn.Linear(mlp_dim, out_dim))

    def forward(self, x):
        return self.fc_hid_out(self.fc_dec_mlp(x))


class CrossSectionalStandardizeTest(unittest.TestCase):
    def test_statistics_modes_permutation_and_autograd(self):
        torch.manual_seed(20)
        layer = Layer.CrossSectionalStandardize()
        x = (torch.randn(20, 4, dtype=torch.float64) * torch.arange(1, 5) + 3).requires_grad_()
        out = layer(x)
        var = x.var(0, unbiased=False)
        torch.testing.assert_close(out.mean(0), torch.zeros(4, dtype=x.dtype), atol=1e-12, rtol=0)
        torch.testing.assert_close(out.var(0, unbiased=False), var / (var + layer.eps))
        torch.testing.assert_close(layer.eval()(x), out, atol=0, rtol=0)
        order = torch.randperm(len(x))
        torch.testing.assert_close(layer(x[order]), out[order])
        self.assertEqual(dict(layer.named_parameters()), {})
        self.assertEqual(dict(layer.named_buffers()), {})
        self.assertTrue(torch.autograd.gradcheck(layer, (x,)))

    def test_constant_singleton_and_low_precision(self):
        layer = Layer.CrossSectionalStandardize()
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for rows in (1, 12):
                with self.subTest(dtype=dtype, rows=rows):
                    x = torch.randn(rows, 3).to(dtype)
                    x[:, 0] = 7
                    x.requires_grad_()
                    out = layer(x)
                    self.assertEqual(out.dtype, dtype)
                    torch.testing.assert_close(out[:, 0], torch.zeros_like(out[:, 0]))
                    if rows == 1:
                        torch.testing.assert_close(out, torch.zeros_like(out))
                    if dtype in (torch.float16, torch.bfloat16):
                        torch.testing.assert_close(out, layer(x.float()).to(dtype), atol=0, rtol=0)
                    (out * torch.randn_like(out)).sum().backward()
                    self.assertTrue(torch.isfinite(out).all())
                    self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):
            layer(torch.zeros(2, 3, 4))


class RNNHiddenStandardizeTest(unittest.TestCase):
    def test_default_checkpoint_and_numerical_compatibility(self):
        for enabled in (False, True):
            kwargs = dict(input_dim=3, hidden_dim=4, dec_mlp_dim=6,
                          dropout=0, hidden_as_factors=enabled)
            torch.manual_seed(10)
            with patch.object(RNN, 'uni_rnn_decoder', LegacyDecoder):
                legacy = RNN.gru(**kwargs)
            torch.manual_seed(10)
            current = RNN.gru(**kwargs)
            self.assertEqual(tuple(current.state_dict()), tuple(legacy.state_dict()))
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(current.state_dict()[key], value, atol=0, rtol=0)
            current.load_state_dict(legacy.state_dict(), strict=True)
            x = torch.randn(12, 5, 3)
            for training in (True, False):
                legacy.train(training)
                current.train(training)
                old_pred, old_other = legacy(x)
                pred, other = current(x)
                torch.testing.assert_close(pred, old_pred, atol=0, rtol=0)
                torch.testing.assert_close(other['hidden'], old_other['hidden'], atol=0, rtol=0)

    def test_std_models_forward_backward_and_hidden_eval(self):
        for multi in (False, True):
            for heads in (1, 2):
                with self.subTest(multi=multi, heads=heads):
                    factory = RNN.rnn_multivariate if multi else RNN.gru
                    model = factory(input_dim=[3, 2] if multi else 3, hidden_dim=4,
                                    act_type='leaky', rnn_type='gru', dropout=0,
                                    hidden_as_factors=True, hidden_norm='std', num_output=heads)
                    x = [torch.randn(12, 5, 3), torch.randn(12, 5, 2)] if multi else torch.randn(12, 5, 3)
                    pred, other = model(x)
                    self.assertEqual(pred.shape, (12, heads))
                    self.assertEqual(other['hidden'].shape, (12, 4))
                    (pred * torch.randn_like(pred)).sum().backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in model.encoder.parameters()))
                    torch.testing.assert_close(model.eval()(x)[1]['hidden'], other['hidden'])
                    self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in model.decoder.modules()))
                    self.assertEqual(sum(isinstance(m, Layer.CrossSectionalStandardize)
                                         for m in model.decoder.modules()), heads)
                    self.assertEqual(sum(isinstance(m, nn.BatchNorm1d)
                                         for m in model.mapping.modules()), heads)

    def test_disabled_normalization_and_scalar_decoder(self):
        for mode in ('batch', 'std'):
            model = RNN.gru(input_dim=3, hidden_dim=4, hidden_as_factors=False, hidden_norm=mode)
            self.assertTrue(all(isinstance(m.fc_hid_out, nn.Linear) for m in model.decoder.mod_list))
        decoder = RNN.uni_rnn_decoder(4, 'leaky', 1, None, 0, True,
                                      map_to_one=True, hidden_norm='std')
        self.assertEqual(decoder(torch.randn(12, 4)).shape, (12, 1))
        with self.assertRaises(ValueError):
            RNN.gru(input_dim=3, hidden_dim=4, hidden_norm='typo')

    def test_schedule_loading_and_baseline_equivalence(self):
        root = Path(__file__).resolve().parents[1] / 'configs/schedule/current'
        baseline = yaml.safe_load((root / 'gru_day_new.yaml').read_text())
        new = yaml.safe_load((root / 'gru_day_new_std.yaml').read_text())
        expected = copy.deepcopy(baseline)
        expected['train'].update({'dataloader.sample_method': 'sequential',
                                  'dataloader.shuffle_option': 'epoch'})
        expected['algo']['gru'].update(hidden_norm=['std'], output_as_factors=[True])
        self.assertEqual(new, expected)
        schedule = ScheduleConfig(schedule_name='gru_day_new_std', vb_level='never')
        self.assertEqual(schedule['model.name'], 'gru_day_new_std')
        algo = AlgoConfig(None, start_with_none=True, module='gru',
                          schedule_config=schedule, vb_level='never').expand()
        self.assertEqual(algo.n_model, 1)
        params = algo.params[0]
        self.assertEqual(params['hidden_norm'], 'std')
        model = RNN.gru(input_dim=3, **params)
        self.assertIsInstance(model.decoder.mod_list[0].fc_hid_out[1], Layer.CrossSectionalStandardize)
        self.assertIsInstance(model.mapping.mod_list[0].fc_map_out[-1], nn.BatchNorm1d)


if __name__ == '__main__':
    unittest.main()
