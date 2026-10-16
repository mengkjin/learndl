"""ResNet block behavior, configuration propagation and input contracts."""
from pathlib import Path
import unittest

import torch
from torch import nn
import yaml

from src.res.algo.nn.api import get_nn_module
from src.res.algo.nn.model.CNN import (
    _resnet_block_1d, _resnet_block_2d, mod_resnet_1d, mod_resnet_2d,
)
from src.res.algo.nn.model.RNN import resnet_gru as legacy_resnet_gru
from src.res.model.util.config.config import AlgoConfig, ScheduleConfig


class ResNetTest(unittest.TestCase):
    def test_forward_backward_and_noncontiguous_inputs(self):
        for factory in (mod_resnet_1d, mod_resnet_2d):
            for width in (1, 3, 16):
                with self.subTest(factory=factory.__name__, width=width):
                    model = factory(8, 5, width, dim_med=7, act_type='gelu', resnet_blocks=2)
                    x = torch.randn(2, 3, 5, 8).transpose(-1, -2).requires_grad_()
                    for training in (True, False):
                        model.train(training)
                        y = model(x)
                        self.assertEqual(y.shape, (2, 3, width))
                        y.square().mean().backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        for name, p in model.named_parameters():
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                    for block in model.blocks:
                        self.assertEqual(block.conv[0].out_channels, 7)
                        self.assertIsInstance(block.conv[2], nn.GELU)
                        self.assertIsInstance(block.act, nn.GELU)
                    head_width = 8 * width * (5 if factory is mod_resnet_2d else 1)
                    self.assertEqual(model.fc_out.in_features, head_width)

    def test_dynamic_default_width_and_positional_compatibility(self):
        for factory in (mod_resnet_1d, mod_resnet_2d):
            for width in (1, 3, 16, 32):
                model = factory(8, 5, width, None, 2)
                self.assertEqual(model.dim_med, max(1, width // 4))
                self.assertEqual(len(model.blocks), 2)
                self.assertIsNone(model.blocks[0].clip_value)
        for factory in (_resnet_block_1d, _resnet_block_2d):
            self.assertEqual(factory(5, 32).dim_med, 8)

    def test_activation_happens_after_residual_addition(self):
        for factory, shape in ((_resnet_block_1d, (2, 4, 5)),
                               (_resnet_block_2d, (2, 4, 5, 3))):
            block = factory(4, 4, act_type='relu').eval()
            with torch.no_grad():
                for p in block.conv.parameters():
                    p.zero_()
                block.conv[-1].bias.fill_(-3)
            # The negative branch must cancel the positive shortcut before ReLU.
            torch.testing.assert_close(block(torch.full(shape, 2.)), torch.zeros(shape))
            torch.testing.assert_close(block(torch.full(shape, 5.)), torch.full(shape, 2.))

    def test_shortcut_is_unclipped_by_default_and_clipping_is_opt_in(self):
        for factory, shape in ((_resnet_block_1d, (2, 4, 5)),
                               (_resnet_block_2d, (2, 4, 5, 3))):
            for clip in (None, 10.):
                block = factory(4, 4, clip_value=clip, act_type='relu').eval()
                with torch.no_grad():
                    for p in block.conv.parameters():
                        p.zero_()
                x = torch.full(shape, 20., requires_grad=True)
                y = block(x)
                torch.testing.assert_close(y, torch.full(shape, 20. if clip is None else 10.))
                y.sum().backward()
                torch.testing.assert_close(x.grad, torch.full(shape, 1. if clip is None else 0.))

    def test_invalid_parameters_fail_at_construction(self):
        for factory in (mod_resnet_1d, mod_resnet_2d):
            for key in ('seq_len', 'feat_len', 'dim_out', 'dim_med', 'resnet_blocks'):
                for value in (0, -1, 1.5, True, '4'):
                    with self.subTest(factory=factory.__name__, key=key, value=value):
                        kwargs = dict(seq_len=8, feat_len=5, dim_out=16)
                        kwargs[key] = value
                        with self.assertRaisesRegex(ValueError, key):
                            factory(**kwargs)
            for value in (0, -1, float('nan'), float('inf'), True, '10'):
                with self.assertRaisesRegex(ValueError, 'clip_value'):
                    factory(8, 5, clip_value=value)
            with self.assertRaises(KeyError):
                factory(8, 5, act_type='not-an-activation')
        for factory in (_resnet_block_1d, _resnet_block_2d):
            for key in ('dim_in', 'dim_out', 'dim_med'):
                with self.assertRaisesRegex(ValueError, key):
                    factory(**(dict(dim_in=4) | {key: 0}))

    def test_input_contracts_reject_incorrect_axes_and_empty_dimensions(self):
        for factory in (mod_resnet_1d, mod_resnet_2d):
            model = factory(8, 5, 16)
            for shape in ((2, 8, 5), (2, 3, 5, 8), (2, 3, 7, 5),
                          (2, 3, 8, 4), (0, 3, 8, 5), (2, 0, 8, 5)):
                with self.assertRaisesRegex(ValueError, 'shape'):
                    model(torch.randn(shape))
            with self.assertRaisesRegex(TypeError, 'floating point'):
                model(torch.ones(2, 3, 8, 5, dtype=torch.int64))
            with self.assertRaisesRegex(TypeError, 'Tensor'):
                model([1, 2])
        for factory, shapes in (
            (_resnet_block_1d, ((2, 8, 5), (2, 5, 8, 1))),
            (_resnet_block_2d, ((2, 8, 5), (2, 2, 8, 5))),
        ):
            block = factory(5 if factory is _resnet_block_1d else 1)
            for shape in shapes:
                with self.assertRaises(ValueError):
                    block(torch.randn(shape))

    def test_model_entrypoints_propagate_resnet_options(self):
        kwargs = dict(input_dim=5, hidden_dim=16, enc_in_dim=8, inday_dim=8,
                      seq_len=3, dim_med=6, act_type='gelu', resnet_blocks=2, dropout=0.)
        for name in ('resnet_gru', 'resnet_mlp', 'resnet_transformer', 'legacy_resnet_gru'):
            with self.subTest(name=name):
                factory = legacy_resnet_gru if name == 'legacy_resnet_gru' else get_nn_module(name)
                model = factory(**kwargs)
                encoder = (model.encoder.mod_list[0].fc_enc_in if name == 'legacy_resnet_gru'
                           else model.fc_enc_in)
                for block in encoder.blocks:
                    self.assertEqual(block.dim_med, 6)
                    self.assertIsInstance(block.act, nn.GELU)
                y, _ = model(torch.randn(4, 3, 8, 5))
                self.assertEqual(y.shape, (4, 1))
                y.square().mean().backward()

    def test_graph_projector_propagates_activation(self):
        for name in ('astgnn', 'astgnn_legacy'):
            model = get_nn_module(name)(input_dim=[5, 3], hidden_dim=16, enc_in_dim=8,
                                        ab_split_input=True, resnet_projector=True,
                                        inday_dim=8, dim_med=6, act_type='gelu')
            for block in model.alpha_fc_enc_in.blocks:
                self.assertEqual(block.dim_med, 6)
                self.assertIsInstance(block.act, nn.GELU)

    def test_config_defaults_and_schedule_override(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/algo/nn/resnet_gru.yaml').read_text())
        self.assertEqual(config['dim_med'], [None])
        self.assertEqual(config['clip_value'], [None])
        schedule = ScheduleConfig(schedule_name='gru_15m_new', vb_level='never')
        algo = AlgoConfig(None, start_with_none=True, module='resnet_gru',
                          schedule_config=schedule, vb_level='never').expand()
        params = dict(algo.params[0])
        self.assertIsNone(params['dim_med'])
        params.update(dim_med=11, act_type='gelu', inday_dim=8)
        model = get_nn_module('resnet_gru')(input_dim=5, **params)
        self.assertEqual(model.fc_enc_in.blocks[0].dim_med, 11)
        self.assertIsInstance(model.fc_enc_in.blocks[0].act, nn.GELU)


if __name__ == '__main__':
    unittest.main()
