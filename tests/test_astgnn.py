"""ASTGNN factor shape and training regression tests on synthetic CPU inputs."""
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch
import io
import numpy as np
import unittest

import torch

from src.res.algo.nn.model.gnn import Astgnn, AstgnnLegacy, _AsymmetricStockGraph
from src.res.algo.nn.api import get_nn_module
from src.res.model.util.config import ModelConfig
from src.res.model.util.data.operations import DataOperator
from src.res.model.util.core.batch import BatchInput, BatchOutput, BatchData
from src.res.model.util.metric.functions import LossFunction


class AstgnnTest(unittest.TestCase):
    def test_schedule_paths_and_head_options(self):
        for path, normalized, mean_pool, beta_pred in product(
            ('day', '15m', 'mincr'), (False, True), (False, True), (False, True)
        ):
            with self.subTest(path=path, hidden_as_factors=normalized,
                              hidden_mean_pool=mean_pool, beta_into_pred=beta_pred):
                torch.manual_seed(42)
                batch, steps = 96, 4
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
                    if name.startswith('beta_map_out.'):
                        # Output-only option: the specified fit objective is alpha-only.
                        self.assertIsNone(parameter.grad, name)
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

                before = model.beta_net[1].value.weight.detach().clone()
                torch.optim.Adam(model.parameters(), lr=1e-3).step()
                self.assertFalse(torch.equal(before, model.beta_net[1].value.weight))
                model.eval()
                with torch.no_grad():
                    prediction, _ = model(inputs)
                self.assertEqual(prediction.shape, (batch, 1))
                self.assertTrue(torch.isfinite(prediction).all().item())



class AstgnnGraphTest(unittest.TestCase):
    def test_adjacency_and_permutation(self):
        torch.manual_seed(17)
        graph = _AsymmetricStockGraph(4).double()
        x = torch.randn(12, 4, dtype=torch.double)
        a = graph.adjacency(x)
        torch.testing.assert_close(a.diag(), torch.zeros(12, dtype=x.dtype))
        torch.testing.assert_close(a.sum(1), torch.ones(12, dtype=x.dtype))
        self.assertFalse(torch.allclose(a, a.T))
        perm = torch.randperm(len(x))
        torch.testing.assert_close(graph(x[perm]), graph(x)[perm])
        single = x[:1]
        torch.testing.assert_close(graph(single), graph.output(single))

    def test_other_stocks_affect_beta_only_and_both_endpoints_use_graph(self):
        torch.manual_seed(23)
        model = Astgnn(3, hidden_dim=8, alpha_num=3, beta_num=2, dropout=0).eval()
        x = torch.randn(12, 4, 3)
        seen = []
        handle = model.beta_net[1].register_forward_hook(lambda m, args, out: seen.append(out))
        _, factors = model(x)
        handle.remove()
        self.assertEqual(len(seen), 2)
        changed = x.clone()
        changed[1:] += 20
        _, other = model(changed)
        torch.testing.assert_close(factors['alphas'][0], other['alphas'][0])
        self.assertGreater((factors['betas'][0] - other['betas'][0]).abs().max().item(), 1e-6)
        self.assertGreater((factors['betas_peer'][0] - other['betas_peer'][0]).abs().max().item(), 1e-6)
        perm = torch.randperm(len(x))
        pred, shuffled = model(x[perm])
        original, _ = model(x)
        torch.testing.assert_close(pred, original[perm])
        for key in ('alphas', 'betas', 'betas_peer'):
            torch.testing.assert_close(shuffled[key], factors[key][perm])

    def test_single_stock_training_and_single_step_turnover(self):
        for normalized, pred_bn in product((False, True), repeat=2):
            model = Astgnn(3, alpha_num=3, beta_num=2, dropout=0,
                           hidden_as_factors=normalized, pred_batchnorm=pred_bn)
            pred, factors = model(torch.randn(1, 1, 3))
            torch.testing.assert_close(factors['betas'], factors['betas_peer'])
            loss = sum(model.loss(pred, torch.randn(1, 2), **factors).values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())


class AstgnnLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = Astgnn(3, alpha_num=3, beta_num=2)

    def test_projection_and_alpha_explanatory_power(self):
        f = torch.randn(20, 5, dtype=torch.double, requires_grad=True)
        y = torch.randn(20, dtype=torch.double)
        projection = f @ torch.linalg.solve(f.T @ f, f.T @ y)
        reference = (y - projection).square().sum() / y.square().sum()
        actual = self.model.rsquare_loss(f, y)
        torch.testing.assert_close(actual, reference)
        self.assertTrue(torch.autograd.gradcheck(self.model.rsquare_loss, (f, y)))
        betas = torch.eye(4, dtype=torch.double)[:, :1]
        alpha = torch.eye(4, dtype=torch.double)[:, 1:2]
        y = alpha[:, 0]
        self.assertAlmostEqual(self.model.rsquare_loss(betas, y).item(), 1.)
        self.assertAlmostEqual(self.model.rsquare_loss(torch.cat((alpha, betas), 1), y).item(), 0.)

    def test_degenerate_factors_and_labels_have_finite_gradients(self):
        base = torch.randn(8, 1, dtype=torch.double)
        for f in (base.repeat(1, 4), torch.zeros(8, 3, dtype=torch.double),
                  torch.ones(8, 3, dtype=torch.double), torch.randn(3, 8, dtype=torch.double),
                  torch.zeros(1, 3, dtype=torch.double)):
            for zero_label in (False, True):
                with self.subTest(shape=f.shape, zero_label=zero_label):
                    h = f.clone().requires_grad_()
                    y = torch.zeros(len(h), dtype=h.dtype) if zero_label else torch.randn(len(h), dtype=h.dtype)
                    r2 = self.model.rsquare_loss(h, y)
                    corr = self.model.corr_loss(h)
                    self.assertTrue(torch.isfinite(r2 + corr))
                    if zero_label:
                        self.assertEqual(r2.item(), 0.)
                    (r2 + corr).backward()
                    self.assertTrue(torch.isfinite(h.grad).all())

    def test_correlations_and_turnover(self):
        h = torch.randn(20, 5, dtype=torch.double)
        torch.testing.assert_close(self.model.corr_loss(h), torch.corrcoef(h.T).norm())
        self.assertEqual(self.model.corr_loss(torch.ones(5, 3)).item(), 0.)
        self.assertEqual(self.model.turnover_loss(h, h).item(), 0.)
        self.assertAlmostEqual(self.model.turnover_loss(h, h + 2).item(), 4.)

    def test_fit_uses_alpha_and_independent_coefficients(self):
        model = Astgnn(3, alpha_num=3, beta_num=2, loss_fit_weight=2,
                       loss_rsquare_weight=3, loss_corr_lamb=.2, loss_turnover_lamb=.7)
        pred, outputs = model(torch.randn(12, 4, 3))
        labels = torch.randn(12, 2)
        weights = torch.rand(12, 2) + .2
        values = model.loss(pred + 100, labels, **outputs, weight=weights)
        expected = ((outputs['pred_alpha'] - labels[:, :1]).square() *
                    weights[:, :1] / weights[:, :1].mean()).mean()
        torch.testing.assert_close(values['mse'], 2 * expected)
        f = torch.cat((outputs['alphas'], outputs['betas']), 1)
        torch.testing.assert_close(values['rsquare'], 3 * model.rsquare_loss(f, labels[:, 1]))
        torch.testing.assert_close(values['corr'], .2 * model.corr_loss(f))
        torch.testing.assert_close(values['turnover'], .7 * model.turnover_loss(outputs['betas'], outputs['betas_peer']))
        for fit_loss in ('ccc', 'pearson'):
            net = Astgnn(3, alpha_num=3, beta_num=2, fit_loss=fit_loss)
            pred, outputs = net(torch.randn(12, 4, 3))
            loss = sum(net.loss(pred, labels, **outputs).values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_training_loss_dispatch(self):
        x, y = torch.randn(24, 4, 3), torch.randn(24, 2)
        batch = BatchInput(x, y, None, torch.zeros(24, 2, dtype=torch.long),
                           torch.ones(24, dtype=torch.bool), np.array([20240101]), np.arange(24))
        data = BatchData(batch, BatchOutput(self.model(x)))
        fn = LossFunction({'ccc': {}}, self.model)
        actual = fn.losses(data, dataset='train')
        self.assertEqual(set(actual), {'mse', 'rsquare', 'corr', 'turnover'})
        sum(actual.values()).backward()
        self.assertIsNotNone(self.model.beta_net[1].query.weight.grad)


class AstgnnConfigTest(unittest.TestCase):
    def config(self, schedule='astgnn_day', **override):
        return ModelConfig(schedule_name=schedule, override=override, short_test=False,
                           stage=1, resume=0, selection=0, vb_level='never')

    def test_schedules_and_legacy_defaults(self):
        with patch('builtins.input', side_effect=AssertionError('No prompts')):
            for path in ('day', '15m', 'mincr'):
                config = self.config(f'astgnn_{path}')
                self.assertEqual(config.labels, ['rtn_lag1_5', 'rtn_lag1_20'])
                self.assertEqual(config.label_normalization, 'raw')
                self.assertEqual(config.skip_horizon, 22)
                self.assertEqual(config.sample_method, 'sequential')
                self.assertEqual(config.algo_config.params[0]['fit_loss'], 'mse')
                self.assertFalse(config.algo_config.params[0]['pred_batchnorm'])
                legacy = self.config(f'astgnn_legacy_{path}')
                self.assertEqual(legacy.model_module, 'astgnn_legacy')
                self.assertEqual(legacy.label_normalization, 'standardize')
                self.assertEqual(legacy.labels, ['std_lag1_10', 'rtn_lag1_10'])
                self.assertEqual(legacy.algo_config.params[0]['fit_loss'], 'ccc')
            config = self.config(**{'model.labels': ['rtn_lag1_10', 'rtn_lag1_5']})
            self.assertEqual(config.labels, ['rtn_lag1_10', 'rtn_lag1_5'])

    def test_invalid_sampling_horizon_and_normalization(self):
        for override, message in (
            ({'train.dataloader.sample_method': 'train_shuffle'}, 'sequential'),
            ({'train.dataloader.sample_method': 'both_shuffle'}, 'sequential'),
            ({'train.skip_horizon': 21}, 'strictly greater'),
            ({'model.labels': ['rtn_lag1_5']}, 'two labels'),
            ({'train.label_normalization': 'other'}, 'normalization'),
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                self.config(**override)

    def test_both_label_preparation_stages_and_standard_default(self):
        labels = torch.tensor([[[.01, .03], [.07, .02]], [[.02, -.01], [.04, -.02]], [[-.03, .05], [.01, .09]]])
        config = SimpleNamespace(label_normalization='raw', weight_scheme=lambda *a: 'equal')
        operator = DataOperator(config)
        first, _ = operator.standardize_y(labels, no_weight=True)
        effective = torch.tensor([[True, True], [True, False], [True, True]])
        second, _ = operator.standardize_y(first, effective[:, 1:], torch.tensor([1]))
        torch.testing.assert_close(first, labels)
        torch.testing.assert_close(second[:, 0][effective[:, 1]], labels[:, 1][effective[:, 1]])
        self.assertTrue(torch.isnan(second[1, 0]).all())
        torch.testing.assert_close(labels[1, 1], torch.tensor([.04, -.02]))
        del config.label_normalization
        standardized, _ = operator.standardize_y(labels)
        torch.testing.assert_close(standardized.mean(0), torch.zeros(2, 2), atol=1e-6, rtol=0)
        self.assertFalse(torch.allclose(standardized, labels))


class AstgnnCheckpointTest(unittest.TestCase):
    def test_legacy_keys_and_new_checkpoint_roundtrip(self):
        self.assertIs(get_nn_module('astgnn_legacy'), AstgnnLegacy)
        self.assertIs(get_nn_module('astgnn'), Astgnn)
        legacy = AstgnnLegacy(3)
        legacy_copy = AstgnnLegacy(3)
        legacy_copy.load_state_dict(legacy.state_dict(), strict=True)
        self.assertIn('beta_net.1.running_mean', legacy.state_dict())
        self.assertNotIn('beta_net.1.query.weight', legacy.state_dict())
        model = Astgnn(3)
        for strict in (True, False):
            with self.assertRaisesRegex(RuntimeError, 'astgnn_legacy'):
                model.load_state_dict(legacy.state_dict(), strict=strict)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        copy = Astgnn(3)
        copy.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        model.eval()
        copy.eval()
        x = torch.randn(12, 4, 3)
        torch.testing.assert_close(model(x)[0], copy(x)[0])
        legacy.eval()
        legacy_copy.eval()
        torch.testing.assert_close(legacy(x)[0], legacy_copy(x)[0])


if __name__ == '__main__':
    unittest.main()
