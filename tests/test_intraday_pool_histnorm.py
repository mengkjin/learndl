import unittest
import torch
from src.res.algo.nn.model.CNN import mod_resnet_1d, mod_resnet_2d, _IntradayPool
from src.res.model.util.data.prenorm import SingleDataPrenorm, SpecialDataPrenorm
from src.data.preprocess.hist_norm import PreProHistNorm


class IntradayTests(unittest.TestCase):
    def test_pool_shapes_gradients(self):
        for factory in (mod_resnet_1d, mod_resnet_2d):
            for mode in ('flatten', 'mean', 'linear', 'attention', 'conv'):
                for length in (1, 7, 16):
                    model = factory(length, 3, 8, resnet_pool=mode, resnet_pool_conv_layers=2)
                    x = torch.randn(2, 4, 3, length).transpose(-1, -2).requires_grad_()
                    y = model(x)
                    self.assertEqual(y.shape, (2, 4, 8))
                    y.square().mean().backward()
                    assert x.grad is not None
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_pool_semantics(self):
        x = torch.randn(3, 8, 16)
        linear = _IntradayPool(16, 8, 'linear')
        torch.testing.assert_close(linear(x), x.mean(-1))
        att = _IntradayPool(16, 8, 'attention').eval()
        torch.testing.assert_close(att.weights(x).sum(-1), torch.ones(3))
        torch.testing.assert_close(att(x)[:1], att(x[:1]))
        self.assertEqual(sum(p.numel() for p in linear.parameters()), 17)

    def test_prenorm_validation_and_broadcast(self):
        op = SingleDataPrenorm.from_input('15m', {'histnorm': True})
        x = torch.randn(2, 3, 16, 6)
        norm = PreProHistNorm(torch.randn(16, 6), torch.ones(16, 6))
        torch.testing.assert_close(op(x, norm), (x - norm.avg) / (norm.std + 1e-6))
        for bad in (None, PreProHistNorm(torch.zeros(15, 6), torch.ones(15, 6)),
                    PreProHistNorm(torch.zeros(16, 6), -torch.ones(16, 6))):
            with self.assertRaises(ValueError):
                op(x, bad)

    def test_30m_denominator_and_no_mutation(self):
        for days in (1, 3, 10):
            x = torch.rand(2, days, 8, 2) + 1
            original = x.clone()
            flat = x.flatten(1, 2)
            divisor = flat[:, -1:].clone()
            divisor[..., 1] = x[:, -1, :, 1].sum(-1, keepdim=True)
            expected = flat / (divisor + 1e-6)
            actual = SpecialDataPrenorm.sp_30mcont(x, features=['close', 'turn_fl'])
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x, original)
            norm = PreProHistNorm(torch.zeros(days * 8, 2), torch.ones(days * 8, 2))
            op = SingleDataPrenorm('30mcont', divlast=True, histnorm=True)
            normalized = op(x, norm, ['close', 'turn_fl'])
            self.assertTrue((normalized[:, -1, 0] == 0).all())
            self.assertTrue((normalized[:, -1, 1] > 0).all())
            off = SingleDataPrenorm('30mcont')
            torch.testing.assert_close(off(x, norm, ['close', 'turn_fl']), flat)

    def test_statistics_and_cutoff(self):
        import numpy as np
        from unittest.mock import patch
        from src.data.util import DataBlock
        values = torch.rand(3, 90, 16, 2) + 1
        dates = np.arange(20160101, 20160191)
        with patch.object(PreProHistNorm, 'save'):
            norm = PreProHistNorm.calculate(DataBlock(values, np.arange(3), dates, ['close', 'turn_fl']), '15m')
            assert norm is not None
            ref = values[:, np.arange(4, 90, 5)].flatten(0, 1)
            torch.testing.assert_close(norm.avg, ref.mean(0))
            torch.testing.assert_close(norm.std, ref.std(0))
            future = torch.cat([values, torch.full((3, 5, 16, 2), 1e6)], 1)
            later = PreProHistNorm.calculate(DataBlock(future, np.arange(3), np.r_[dates, np.arange(20170101, 20170106)], ['close', 'turn_fl']), '15m')
            assert later is not None
            torch.testing.assert_close(later.avg, norm.avg)
            torch.testing.assert_close(later.std, norm.std)

    def test_schedule_expansion(self):
        from src.res.model.util.config.config import AlgoConfig, ScheduleConfig
        pools = []
        for name in ('resnet_gru_new_15m_pool', 'resnet_gru_new_15m_pool_conv2'):
            schedule = ScheduleConfig(schedule_name=name, vb_level='never')
            algo = AlgoConfig(None, start_with_none=True, module='resnet_gru_new', schedule_config=schedule, vb_level='never').expand()
            pools.extend((p['resnet_pool'], p['resnet_pool_conv_layers']) for p in algo.params)
        self.assertEqual(pools, [('flatten', 1), ('mean', 1), ('linear', 1), ('attention', 1), ('conv', 1), ('conv', 2)])
