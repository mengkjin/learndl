"""Source-qualified labels: alignment, provenance and training compatibility."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import DataBlock
from src.data.preprocess import PrePros
from src.data.preprocess.core import PreProcessor
from src.data.preprocess.processors import PrePro_y, PrePro_y_lag0
from src.data.preprocess.task import DATASET_FIT, DATASET_PREDICT
from src.res.model.util.label_spec import LabelSpec, parse_labels
from src.res.model.util.config import ModelConfig
from src.res.model.util.data.module_data import ModuleData
from src.res.model.util.data.data_module import DataModule
from src.res.model.util.data.operations import DataOperator

DATES = np.array([20240102, 20240103, 20240104])


def block(secids, dates, features, values=None, inday=1):
    shape = (len(secids), len(dates), inday, len(features))
    values = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) + 1 if values is None else torch.tensor(values, dtype=torch.float32).reshape(shape)
    return DataBlock(values, np.array(secids), np.array(dates), np.array(features))


class LabelSourceTest(TestCase):
    def load(self, labels, sources, **kwargs):
        data = ModuleData.initialize(['day'], labels, vb_level='never', **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(ModuleData, 'target_start_end', return_value=(DATES[0], DATES[-1])))
            stack.enter_context(patch('src.res.model.util.data.module_data.CALENDAR.range', return_value=DATES))
            stack.enter_context(patch.object(ModuleData, '_load_norms'))
            loader = stack.enter_context(patch.object(data, 'load_one', side_effect=lambda key, **kw: sources[key].copy()))
            data.load()
        return data, [c.args[0] for c in loader.call_args_list]

    def test_parsing_and_duplicates(self):
        self.assertEqual(LabelSpec.parse('y@a'), LabelSpec('y', 'a'))
        self.assertEqual([s.name for s in parse_labels(['y@a', 'y_lag0@a'])], ['a', 'y_lag0@a'])
        for bad in ['', '@a', 'y@', 'y@a@b', ' y@a', 'a ', None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LabelSpec.parse(bad)
        for bad in [[], 'a', ['a', 'y@a'], ['y_lag0@a', 'y_lag0@a']]:
            with self.assertRaises(ValueError):
                parse_labels(bad)

    def test_union_order_values_and_no_label_ffill(self):
        sources = {
            'y': block([1, 2], DATES[:2], ['a'], [11, 12, 21, 22]),
            'y_lag0': block([2, 3], DATES[1:], ['a', 'b'], [201, 211, 202, 212, 301, 311, 302, 312]),
            'day': block([1, 2, 3], DATES, ['close']),
        }
        data, calls = self.load(['y_lag0@b', 'a', 'y_lag0@a'], sources)
        self.assertEqual(calls, ['y_lag0', 'y', 'day'])
        self.assertEqual(list(data.y.feature), ['y_lag0@b', 'a', 'y_lag0@a'])
        np.testing.assert_array_equal(data.secid, [1, 2, 3])
        np.testing.assert_array_equal(data.date, DATES)
        self.assertEqual(set(data.x), {'day'})
        self.assertEqual(set(data.blocks), {'y', 'day'})
        torch.testing.assert_close(data.y.values[1, 1, 0], torch.tensor([211., 22., 201.]))
        self.assertTrue(torch.isnan(data.y.values[0, 2]).all())
        self.assertFalse(data.label_coverage[0, 2].any())
        self.assertTrue(data.label_coverage[1, 1].all())
        np.testing.assert_array_equal(sources['y_lag0'].feature, ['a', 'b'])

    def test_default_all_labels_and_filter(self):
        sources = {'y': block([1, 2], DATES, ['a', 'b']), 'day': block([1, 2], DATES, ['close'])}
        data, _ = self.load(None, sources, filter_secid='first.1')
        np.testing.assert_array_equal(data.secid, [1])
        np.testing.assert_array_equal(data.y.feature, ['a', 'b'])
        torch.testing.assert_close(data.y.values, sources['y'].values[:1])

    def test_lag0_only_and_nan_prediction(self):
        sources = {'y_lag0': block([2, 3], DATES, ['a'], [float('nan')] * 6), 'day': block([2, 3], DATES, ['close'])}
        data, calls = self.load(['y_lag0@a'], sources, use_data='predict')
        self.assertEqual(calls, ['y_lag0', 'day'])
        self.assertTrue(torch.isnan(data.y.values).all())
        self.assertEqual(data.y.shape, (2, 3, 1, 1))
        op = DataOperator(SimpleNamespace(module_type='linear', input_data_types=[], input_final_types=[]))
        self.assertIsNone(op.effective_samples({}, None, torch.arange(3)))

    def test_unknown_missing_duplicate_intraday_and_input_overlap(self):
        with self.assertRaisesRegex(ValueError, 'Unknown label sources'):
            self.load(['unknown@a'], {})
        for source, message in [(block([1], DATES, ['b']), 'missing fields'), (block([1], DATES, ['a', 'a']), 'duplicate features'), (block([1], DATES, ['a'], inday=2), 'inday'), (DataBlock(), 'no data')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load(['y_lag0@a'], {'y_lag0': source})
        with self.assertRaisesRegex(ValueError, 'cannot be model inputs'):
            ModuleData.initialize(['y_lag0'], ['y_lag0@a'])

    def test_config_identity_and_reload(self):
        a = ModuleData.initialize([], ['y@a', 'y_lag0@a']).config
        b = ModuleData.initialize([], ['a', 'y_lag0@a']).config
        c = ModuleData.initialize([], ['y_lag0@a', 'a']).config
        self.assertTrue(a.covered_by(b))
        self.assertFalse(a.covered_by(c))
        holder = SimpleNamespace(_use_data='fit', datas=SimpleNamespace(use_data='both', y_labels=['a']), config=SimpleNamespace(labels=['y_lag0@a']))
        self.assertFalse(DataModule.no_need_to_load_data(holder))
        holder.config.labels = ['a']
        self.assertTrue(DataModule.no_need_to_load_data(holder))

    def test_source_aware_date_bounds(self):
        data = ModuleData.initialize([], ['y_lag0@a'])
        with patch.object(DataBlock, 'max_data_date', return_value=20240104) as maximum:
            self.assertEqual(data.target_start_end(), (20070101, 20240104))
            maximum.assert_called_once_with('y_lag0', 'fit')
        with patch.object(DataBlock, 'max_data_date', return_value=None), patch('src.res.model.util.data.module_data.CALENDAR.updated', return_value=20240105):
            self.assertEqual(data.target_start_end()[1], 20240105)
        with patch.object(DataBlock, 'min_data_date', side_effect=lambda key, frame: {'y': 20200102, 'y_lag0': 20190102, 'day': 20190603}[key]), patch('src.res.model.util.data.module_data.CALENDAR.td', side_effect=lambda date, **kw: SimpleNamespace(as_int=lambda: date)):
            self.assertEqual(ModuleData.min_data_date(['day'], y_labels=['a', 'y_lag0@a']), 20190603)

    def test_fill_zero_only_in_source_coverage(self):
        op = DataOperator(SimpleNamespace())
        coverage = torch.tensor([[[True, False]], [[False, True]]])
        y = torch.full((2, 1, 2), torch.nan)
        with patch('src.res.model.util.data.operations.CALENDAR.td', side_effect=lambda date, lag=None: _Date(date)), patch('src.data.loader.RISK.get_res', return_value=pd.DataFrame({'secid': [1, 2]})):
            filled = op.fill_y_zero_no_trade(y, DATES[:1], np.array([1, 2]), ['std_lag1_10', 'y_lag0@std_lag0_10'], coverage)
        self.assertEqual(filled[0, 0, 0], 0)
        self.assertEqual(filled[1, 0, 1], 0)
        self.assertTrue(torch.isnan(filled[0, 0, 1]))
        self.assertTrue(torch.isnan(filled[1, 0, 0]))
        self.assertTrue(torch.isnan(y).all())

    def test_horizons_and_astgnn(self):
        self.assertEqual(DataOperator.parse_label_horizon('y_lag0@std_lag0_10'), (0, 10))
        self.assertEqual(DataOperator.parse_label_horizon('y@rtn_lag1_3_open'), (1, 3))
        self.assertIsNone(DataOperator.parse_label_horizon('custom@score'))
        config = SimpleNamespace(sample_method='sequential', labels=['y_lag0@std_lag0_10', 'res_lag1_20'], skip_horizon=22)
        ModelConfig.check_astgnn_config(config)
        config.skip_horizon = 21
        with self.assertRaisesRegex(ValueError, 'strictly greater'):
            ModelConfig.check_astgnn_config(config)

    def test_fixed_eval_labels_load_only_when_needed(self):
        from src.res.model.callback.test.basic import BasicTestResult
        targets = BasicTestResult.EVAL_LABELS
        data = ModuleData.initialize([], ['y_lag0@a'])
        data.blocks['y'] = block([1], DATES, ['y_lag0@a'])
        evaluation = block([1], DATES, targets)
        callback = SimpleNamespace(data=SimpleNamespace(datas=data), EVAL_LABELS=targets)
        preds = pd.DataFrame({'secid': [1], 'date': [DATES[1]], 'pred': [0.5]})
        with patch.object(data, 'load_one', return_value=evaluation) as loader:
            result = BasicTestResult._attach_eval_labels(callback, preds)
            self.assertEqual(result['rtn_lag1_10'].iloc[0], 3.)
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args, ('y',))
            np.testing.assert_array_equal(loader.call_args.kwargs['dates'], DATES[1:2])
            self.assertEqual(list(data.y.feature), ['y_lag0@a'])
        data.blocks['y'] = evaluation
        with patch.object(data, 'load_one', side_effect=AssertionError('Already loaded')):
            BasicTestResult._attach_eval_labels(callback, preds)

    def test_training_batch_single_and_mixed(self):
        from src.res.algo.nn.api import get_nn_module
        for labels in [['y_lag0@a'], ['y_lag0@a', 'a']]:
            sources = {'y': block([1, 2, 3], DATES, ['a']), 'y_lag0': block([1, 2, 3], DATES, ['a']), 'day': block([1, 2, 3], DATES, ['close'])}
            data, _ = self.load(labels, sources)
            config = SimpleNamespace(module_type='nn', input_data_types=[], input_final_types=[], seq_lens={}, seq_steps={}, label_normalization='standardize', weight_scheme=lambda *args: 'equal')
            op = DataOperator(config)
            y, _ = op.standardize_y(data.y.values.squeeze(2))
            effective = op.effective_samples({}, y, torch.arange(3))
            self.assertTrue(effective.all())
            model = get_nn_module('gru')(input_dim=1, hidden_dim=4, rnn_layers=1, dropout=0., num_output=len(labels))
            x = data.x['day'].values.squeeze(2)
            prediction, _ = model(x)
            target = y[:, -1]
            self.assertEqual(prediction.shape, target.shape)
            loss = (prediction - target).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


class _Date:
    def __init__(self, value): self.value = value
    def __add__(self, lag): return self
    def as_int(self): return self.value


class LabelProcessorTest(TestCase):
    def test_schedule_and_dump_identity(self):
        for schedule in ['gru_day_lag0', 'gru_day_lag0_att']:
            config = ModelConfig(schedule_name=schedule, short_test=False, stage=1,
                                 resume=0, selection=0, vb_level='never')
            self.assertEqual(config.labels, ['y_lag0@std_lag0_10'])
        for frame in ['fit', 'predict']:
            self.assertNotEqual(DataBlock.path_preprocess('y', frame), DataBlock.path_preprocess('y_lag0', frame))
        config = ModelConfig(schedule_name='gru_day_lag0', short_test=False, stage=1,
                             resume=0, selection=0, vb_level='never',
                             override={'model.labels': ['y@std_lag1_10', 'y_lag0@std_lag0_10']})
        self.assertEqual(config['model.labels'], ['std_lag1_10', 'y_lag0@std_lag0_10'])

    def test_registration_and_loaders(self):
        self.assertIs(PreProcessor.registry['y'], PrePro_y)
        self.assertIs(PreProcessor.registry['y_lag0'], PrePro_y_lag0)
        self.assertIn('y_lag0', DATASET_FIT)
        self.assertIn('y_lag0', DATASET_PREDICT)
        self.assertNotIn('_returnlabelpreprocessor', PrePros.keys())
        for cls, keys in [(PrePro_y, ['ret10_lag', 'ret20_lag', 'ret5_lag', 'ret3_lag']), (PrePro_y_lag0, ['ret10', 'ret20', 'ret5'])]:
            with patch('src.data.preprocess.processors.labels.BlockLoader') as loader:
                cls('fit').block_loaders()
                self.assertEqual(loader.call_args_list[0].args, ('labels_ts', keys))

    def test_lag_transforms_equivalent(self):
        torch.manual_seed(0)
        values = torch.randn(12, 3, 1, 2)
        risk = DataBlock(torch.randn(12, 3, 1, 2), np.arange(12), DATES, ['industry', 'size'])
        results = []
        for lag, cls in [(1, PrePro_y), (0, PrePro_y_lag0)]:
            returns = DataBlock(values.clone(), np.arange(12), DATES, [f'rtn_lag{lag}_10', f'res_lag{lag}_10'])
            result = cls('fit').process({'y': returns, 'risk': risk.copy()})
            self.assertEqual(list(result.feature), [f'rtn_lag{lag}_10', f'res_lag{lag}_10', f'std_lag{lag}_10'])
            results.append(result.values)
        torch.testing.assert_close(*results)


if __name__ == '__main__':
    main()
