"""ASTGNN factor shape and training regression tests on synthetic CPU inputs."""
from itertools import product
import unittest

import torch

from src.res.algo.nn.model.gnn import Astgnn


class AstgnnTest(unittest.TestCase):
    def test_schedule_paths_and_head_options(self):
        for path, normalized, mean_pool, beta_pred in product(
            ('day', '15m', 'mincr'), (False, True), (False, True), (False, True)
        ):
            with self.subTest(path=path, hidden_as_factors=normalized,
                              hidden_mean_pool=mean_pool, beta_into_pred=beta_pred):
                torch.manual_seed(42)
                batch, steps = 32, 4
                dims = [4, 3] if path == 'day' else [4, 3, 2]
                inputs = [torch.randn(batch, steps, dim) for dim in dims]
                if path == '15m':
                    inputs[0] = torch.randn(batch, steps, 8, dims[0])
                model = Astgnn(
                    input_dim=dims, ab_split_input=path != 'day',
                    resnet_projector=path == '15m', inday_dim=8,
                    hidden_as_factors=normalized, hidden_mean_pool=mean_pool,
                    beta_into_pred=beta_pred, fit_loss='ccc', dropout=0,
                )
                pred, factors = model(inputs)
                self.assertEqual(pred.shape, (batch, 1))
                self.assertEqual(factors['alphas'].shape, (batch, 60))
                self.assertEqual(factors['betas'].shape, (batch, 10))
                self.assertEqual(factors['betas_peer'].shape, (batch, 10))
                if normalized:
                    for key in ('alphas', 'betas', 'betas_peer'):
                        torch.testing.assert_close(
                            factors[key].mean(0), torch.zeros(factors[key].shape[1]),
                            atol=1e-5, rtol=0,
                        )
                if mean_pool:
                    torch.testing.assert_close(
                        factors['pred_alpha'], factors['alphas'].mean(-1, keepdim=True)
                    )
                if beta_pred:
                    self.assertEqual(factors['pred_beta'].shape, (batch, 1))
                else:
                    self.assertEqual(factors['pred_beta'], 0)

                losses = model.loss(pred, torch.randn(batch, 2), **factors)
                total = sum(losses.values())
                self.assertTrue(torch.isfinite(total).item())
                total.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

                model.eval()
                with torch.no_grad():
                    prediction, _ = model(inputs)
                self.assertEqual(prediction.shape, (batch, 1))
                self.assertTrue(torch.isfinite(prediction).all().item())


if __name__ == '__main__':
    unittest.main()
