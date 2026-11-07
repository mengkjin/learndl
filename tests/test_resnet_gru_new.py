"""Projection widths, registry and backward contract for resnet_gru_new."""
import unittest
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from src.res.algo.nn.api import get_nn_module
from src.res.algo.nn.model.resnet_gru import resnet_gru, resnet_gru_new


class ResnetGRUNewTests(unittest.TestCase):
    def test_independent_widths_and_gradients(self):
        self.assertIs(get_nn_module('resnet_gru_new'), resnet_gru_new)
        for project in (True, 'linear', False, None):
            for attention in (False, True):
                model = resnet_gru_new(6, hidden_dim=12, enc_in=project, enc_in_dim=7,
                                       inday_resnet_dim=9, inday_dim=4, resnet_blocks=1,
                                       enc_att=attention)
                self.assertEqual(model.inday_resnet.dim_in, 7 if project else 6)
                self.assertEqual(model.fc_rnn.gru.input_size, 9)
                y, other = model(torch.randn(3, 5, 4, 6))
                self.assertEqual(y.shape, (3, 1))
                self.assertEqual(other['hidden'].shape, (3, 12))
                (y.square().mean() + other['hidden'].square().mean()).backward()
                for parameter in model.parameters():
                    assert parameter.grad is not None
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_disabled_projection_matches_old_model(self):
        old = resnet_gru(6, hidden_dim=12, enc_in_dim=9, inday_dim=4,
                         resnet_blocks=1, dropout=0).eval()
        new = resnet_gru_new(6, hidden_dim=12, enc_in=False, inday_resnet_dim=9,
                             inday_dim=4, resnet_blocks=1, dropout=0).eval()
        new.inday_resnet.load_state_dict(old.fc_enc_in.state_dict(), strict=True)
        new.fc_rnn.load_state_dict(old.fc_rnn.state_dict(), strict=True)
        new.output.load_state_dict(old.output.state_dict(), strict=True)
        x = torch.randn(3, 5, 4, 6)
        torch.testing.assert_close(new(x), old(x))

    def test_schedule_constructs_registered_model(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/schedule/current/resnet_gru_new_15m.yaml').read_text())
        module = config['model']['module']
        params = {key: value[0] for key, value in config['algo'][module].items()}
        model = get_nn_module(module)(6, inday_dim=16, **params)
        self.assertEqual(model.inday_resnet.dim_in, 16)
        self.assertEqual(model.inday_resnet.dim_out, 32)
        self.assertEqual(model.fc_rnn.gru.input_size, 32)
        self.assertEqual(model.fc_rnn.gru.hidden_size, 64)
        y, _ = model(torch.randn(2, 30, 16, 6))
        self.assertEqual(y.shape, (2, 1))

    def test_invalid_dimensions_and_projection(self):
        for kwargs in ({'enc_in': 'typo'}, {'inday_resnet_dim': 0}, {'rnn_layers': 4},
                       {'enc_in_dim': -1}, {'rnn_type': 'lstm'}):
            with self.assertRaises(ValueError):
                resnet_gru_new(6, inday_dim=16, **cast(dict[str, Any], kwargs))


if __name__ == '__main__':
    unittest.main()
