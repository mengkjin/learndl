"""Legacy/experimental equivalence without loading market data."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.res.model.util.metric.aggregator import MetricAggregator
from src.res.model.util.metric.stage_metrics import BatchMetrics, EpochMetrics, AttemptMetrics
from src.res.model.util.metric.fast_metrics import FastBatchMetrics, FastEpochMetrics, ReusingAccuracyFunction
from src.res.model.util.metric.functions import AccuracyFunction, RankICFunction
from src.res.model.util.core import BatchInput, BatchOutput, BatchData
from src.res.model.callback.monitor.summary import SummaryWriter


def batch_data(pred=None, label=None):
    torch.manual_seed(7)
    if pred is None:
        pred = torch.randn(40, 1)
    if label is None:
        label = torch.randn(40, 1)
    inp = BatchInput(torch.randn(40, 3), label, None, torch.zeros(40, 2, dtype=torch.long),
                     torch.ones(40, 1, dtype=torch.bool), np.array([1]), np.arange(40))
    return BatchData(inp, BatchOutput((pred, {'hidden': torch.randn(40, 4, requires_grad=True)})))


class FastpathTest(TestCase):
    def test_accuracy_reuse_and_fallback(self):
        criterion = {'spearman': {'which_output': [0], 'which_label': [0]}}
        data = batch_data()
        # Include nonfinite inputs and ties, preserving the legacy filtering/ranking.
        data.output.pred[0] = float('inf')
        data.output.pred[1:4] = 0
        old = AccuracyFunction(deepcopy(criterion))
        new = ReusingAccuracyFunction(deepcopy(criterion), model_param={'num_output': 1}, label_count=1)
        self.assertTrue(new.reuse_rankic)
        for dataset in ('train', 'valid'):
            self.assertEqual(old.accuracies(data, dataset=dataset), new.accuracies(data, dataset=dataset))
            torch.testing.assert_close(new.last_rankic, RankICFunction().rankic(data), rtol=0, atol=0)
        self.assertEqual(new.accuracies(data, dataset='test'), {})
        self.assertIsNone(new.last_rankic)
        class Specific(nn.Module):
            def accuracy(self, **kwargs):
                return torch.tensor(0.2)
        for criteria, net, param, labels in [
            ({'global2top': {}}, None, {}, 1),
            (deepcopy(criterion), Specific(), {}, 1),
            (deepcopy(criterion), None, {'num_output': 2}, 1),
            ({'spearman': {'which_label': [1]}}, None, {}, 2),
            ({'spearman': {'lamb': 2}}, None, {}, 1),
        ]:
            fn = ReusingAccuracyFunction(criteria, net, model_param=param, label_count=labels)
            self.assertFalse(fn.reuse_rankic)
        # Preserve immediate NaN accuracy failure.
        bad = batch_data(label=torch.full((40, 1), float('nan')))
        for fn in (old, new):
            with self.assertRaisesRegex(ValueError, 'nan accuracies'):
                fn.accuracies(bad)

    def test_scalar_reads_weights_and_reset(self):
        agg = MetricAggregator()
        batch = FastBatchMetrics(agg)
        batch.new(0)
        loss = torch.tensor(2., requires_grad=True)
        batch.set_values({'a': loss}, {'spearman': 0.2}, torch.tensor(0.2))
        original = torch.Tensor.item
        reads = []
        def item(tensor):
            reads.append(tensor)
            return original(tensor)
        with patch.object(torch.Tensor, 'item', item):
            self.assertEqual(batch.loss_weights, {'a': 1.})
            self.assertEqual(len(reads), 0)
            total = batch.total_loss
            self.assertEqual(len(reads), 1)  # Still before backward.
            total.backward()
            batch.losses_items
            count = len(reads)
            batch.losses_items
            self.assertEqual(len(reads), count)
        agg.inject('loss', {'a': 0.})
        self.assertEqual(batch.loss_weights, {'a': 0.})
        batch.set_metrics('losses', {'a': torch.tensor(3.)})
        self.assertEqual(batch.losses_items, {'a': 3.})
        batch.collect(); batch.close(); batch.new(1)
        batch.set_values({'a': torch.tensor(4.)}, {}, torch.tensor(0.))
        self.assertEqual(batch.losses_items, {'a': 4.})

    def test_complete_tables_partial_reads_nan_and_attempt_interface(self):
        aggs = [MetricAggregator(), MetricAggregator()]
        epochs = [EpochMetrics(aggs[0]), FastEpochMetrics(aggs[1])]
        for ep in epochs:
            ep.new('train', epoch=2, phase=1)
        for i in range(4):
            for agg, ep, cls in zip(aggs, epochs, (BatchMetrics, FastBatchMetrics)):
                agg.inject('loss', {'a': i / 3})
                batch = cls(agg); batch.new(i)
                losses = {'a': torch.tensor(float(i))}
                if i > 1:
                    losses['extra'] = torch.tensor(.5)
                batch.set_values(losses, {'spearman': .2}, torch.tensor(float('nan') if i == 2 else .2))
                _ = batch.total_loss
                ep.append(batch); batch.close()
            for name in ('accuracies', 'losses', 'weights', 'totals'):
                pd.testing.assert_frame_equal(epochs[0].get_table(name), epochs[1].get_table(name))
            self.assertEqual(epochs[0].nanloss, epochs[1].nanloss)
        for ep in epochs:
            ep.collect()
        for name in ('accuracies', 'losses', 'weights', 'totals'):
            pd.testing.assert_frame_equal(epochs[0].get_epoch_table(name), epochs[1].get_epoch_table(name))
        for cls in (EpochMetrics, FastEpochMetrics):
            agg = MetricAggregator(); train = cls(agg); valid = cls(agg)
            for ep, ds in ((train, 'train'), (valid, 'valid')):
                ep.new(ds)
                b = FastBatchMetrics(agg) if cls is FastEpochMetrics else BatchMetrics(agg)
                b.new(0); b.set_values({'a': torch.tensor(1.)}, {'global.spearman': .1, 'top.long_avg': .2}, torch.tensor(.1))
                _ = b.total_loss; ep.append(b); b.close()
            attempt = AttemptMetrics(agg); attempt.new(); attempt.append(train, valid)
            self.assertEqual(pd.concat(attempt.tables['valid_epoch_accuracies']).shape, (1, 2))

    def test_gru_updates_and_gradients_unchanged(self):
        from src.res.algo.nn.model.RNN import gru
        from src.res.model.util.metric.functions import LossFunction
        torch.manual_seed(41)
        models = [gru(input_dim=3, hidden_dim=4, dropout=.1, hidden_as_factors=True)]
        models.append(deepcopy(models[0]))
        opts = [torch.optim.Adam(model.parameters(), lr=.005) for model in models]
        x = torch.randn(40, 5, 3)
        data = batch_data()
        data.input.x = x
        for step in range(3):
            grad_records = []
            values = []
            for idx, (model, opt, cls) in enumerate(zip(models, opts, (BatchMetrics, FastBatchMetrics))):
                torch.manual_seed(100 + step)
                model.train()
                current = BatchData(data.input, BatchOutput(model(x)))
                loss_fn = LossFunction({'ccc_hcorr': {'alpha': .01}}, model)
                criteria = {'spearman': {'which_output': [0], 'which_label': [0]}}
                accuracy_fn = (AccuracyFunction(criteria, model) if idx == 0 else
                               ReusingAccuracyFunction(criteria, model, model_param={'num_output': 1}, label_count=1))
                losses = loss_fn.losses(current)
                accuracies = accuracy_fn.accuracies(current)
                rankic = RankICFunction().rankic(current) if idx == 0 else accuracy_fn.last_rankic
                metrics = cls(MetricAggregator()); metrics.new(step)
                metrics.set_values(losses, accuracies, rankic)
                opt.zero_grad(); metrics.total_loss.backward()
                grad_records.append([p.grad.clone() for p in model.parameters()])
                torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
                opt.step()
                values.append((metrics.total_loss_item, accuracies, rankic.item()))
            self.assertEqual(values[0], values[1])
            for a, b in zip(*grad_records):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)

    def test_empty_test_metrics_and_new_epoch(self):
        for dataset in ('test', 'valid'):
            epochs = []
            for batch_cls, epoch_cls in ((BatchMetrics, EpochMetrics), (FastBatchMetrics, FastEpochMetrics)):
                agg = MetricAggregator(); epoch = epoch_cls(agg); epoch.new(dataset)
                batch = batch_cls(agg); batch.new(0)
                batch.set_values({}, {}, torch.tensor(.1))
                epoch.append(batch); batch.close(); epoch.collect()
                epochs.append(epoch)
            for name in ('accuracies', 'losses', 'weights', 'totals'):
                pd.testing.assert_frame_equal(epochs[0].get_table(name), epochs[1].get_table(name))
            for ep in epochs:
                ep.collected = True; ep.close(); ep.new(dataset, epoch=1)
                self.assertTrue(ep.get_table('totals').empty)

    def test_hidden_diagnostic_only_loses_graph(self):
        data = batch_data()
        records = []
        for enabled in (False, True):
            callback = SimpleNamespace(
                HIDDEN_FEATURE_MODE=True, config={'train.experimental_metrics_fastpath': enabled},
                batch_data=data, metrics=SimpleNamespace(rankic_function=RankICFunction()),
                hidden_rankics=[], hidden_correlations=[])
            SummaryWriter.append_batch_hidden_summary(callback)
            records.append(callback.hidden_correlations[0])
        self.assertTrue(records[0].requires_grad)
        self.assertFalse(records[1].requires_grad)
        self.assertTrue(data.output.hidden.requires_grad)
        torch.testing.assert_close(records[0], records[1], rtol=0, atol=0)


if __name__ == '__main__':
    main()
