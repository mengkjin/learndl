"""Financial identities, sample limits and historical inference isolation."""
from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.res.model.analytic.compare import CompareConfig, CompareModelSpec, CompareResult, compare_models
from src.res.model.analytic.compare.inputs import ModelInput, common_dates, load_inputs
from src.res.model.analytic.compare.statistics import (
    recover_returns, top_stats, ic_stats, periods, correlation_matrix, linear_cka, sample_dates,
)
from src.res.model.analytic.compare.outputs import ArchivedOutputProvider, analyze_outputs, checkpoint_for_date


class CompareStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.Index([20231228, 20231229, 20240102, 20240103], name='date')
        self.daily = pd.DataFrame({'pf': [-.1, .2, .01, -.02], 'bm': [0, .01, 0, .01]}, index=self.dates)
        self.daily['excess'] = self.daily.pf - self.daily.bm
        self.curve = (1 + self.daily[['pf', 'bm']]).cumprod() - 1
        self.curve['excess'] = self.daily.excess.cumsum()

    def test_recover_before_slice_and_zero_baseline(self):
        restored = recover_returns(self.curve, self.dates)
        np.testing.assert_allclose(restored[['pf', 'bm', 'excess']], self.daily, atol=1e-15)
        cut = restored.loc[20240102:]
        stats = top_stats(cut)
        self.assertAlmostEqual(stats['pf_return'], 1.01 * .98 - 1)
        self.assertAlmostEqual(stats['excess_sum'], -.02)
        self.assertAlmostEqual(top_stats(restored.iloc[:1])['excess_mdd'], .1)
        duration = (pd.Timestamp('2024-01-03') - pd.Timestamp('2023-12-29')).days + 1
        self.assertAlmostEqual(stats['excess_annualized'], (1.01 * .97) ** (365 / duration) - 1)

    def test_gap_not_multi_day_return_or_missing_zero(self):
        curve = self.curve.drop(20231229)
        restored = recover_returns(curve, self.dates)
        self.assertTrue(restored.loc[20240102, ['pf', 'bm', 'excess']].isna().all())
        self.assertFalse(top_stats(restored)['complete'])
        self.assertAlmostEqual(top_stats(restored)['pf_return'], self.curve.pf.iloc[-1])
        self.assertTrue(np.isnan(top_stats(restored)['ir']))
        restored = recover_returns(self.curve, self.dates).drop(20231229)
        self.assertFalse(top_stats(restored)['complete'])

    def test_invalid_calendar_and_zero_wealth(self):
        with self.assertRaises(ValueError):
            recover_returns(self.curve, self.dates[:-1])
        curve = self.curve.copy()
        curve.loc[20231228, 'pf'] = -1
        self.assertTrue(np.isnan(recover_returns(curve, self.dates).loc[20231229, 'pf']))

    def test_periods_and_common_dates(self):
        config = CompareConfig(periods=('all', 'year', 'quarter', 'month', 'recent_year'),
                               custom_periods={'event': (20231229, 20240102)})
        groups = periods(self.dates, config)
        self.assertEqual(groups['year:2024'].tolist(), [20240102, 20240103])
        self.assertEqual(groups['custom:event'].tolist(), [20231229, 20240102])
        other = self.daily.drop(20231229)
        self.assertEqual(common_dates([self.daily, other], CompareConfig(start=20231229)).tolist(), [20240102, 20240103])
        with self.assertRaisesRegex(ValueError, 'No common'):
            common_dates([self.daily, other], CompareConfig(start=20250101))

    def test_correlations_use_finite_pairs_and_constants_are_na(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, np.inf], 'b': [2, 4, 6, np.nan, 8], 'c': [1] * 5})
        values, counts, reasons = correlation_matrix(df)
        self.assertAlmostEqual(values.loc['a', 'b'], 1)
        self.assertEqual(counts.loc['a', 'b'], 3)
        self.assertTrue(np.isnan(values.loc['c', 'c']))
        pd.testing.assert_frame_equal(values, values.T)
        self.assertTrue(reasons)
        self.assertEqual(ic_stats(pd.Series([1, 0, np.nan, -1]))['n'], 3)

    def test_cka_rotation_dimensions_and_degenerate(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 3))
        q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        self.assertAlmostEqual(linear_cka(x, x @ q)[0], 1)
        self.assertAlmostEqual(linear_cka(x, np.c_[x, np.zeros((40, 2))])[0], 1)
        self.assertTrue(np.isnan(linear_cka(x, np.ones((40, 2)))[0]))
        self.assertTrue(np.isnan(linear_cka(x[:2], x[:2])[0]))
        self.assertLess(linear_cka(x, rng.normal(size=(40, 5)))[0], .5)

    def test_sampling_and_strict_checkpoint_boundary(self):
        self.assertEqual(sample_dates(range(101), 5), [0, 25, 50, 75, 100])
        self.assertEqual(sample_dates([3, 1, 2, 2], 20), [1, 2, 3])
        self.assertEqual(checkpoint_for_date([20230101, 20240101], 20240101), 20230101)
        with self.assertRaises(ValueError):
            checkpoint_for_date([20240101], 20240101)


class CompareInputTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dates = [20240102, 20240103, 20240104, 20240105]
        self.paths = [self.root / 'a.xlsx', self.root / 'b.xlsx']
        for path in self.paths:
            self.write_report(path)

    def write_report(self, path, duplicate=False):
        ic = pd.DataFrame({'factor_name': ['best', None, None, None], 'benchmark': ['market'] * 4,
                           'date': self.dates, 'ic': [.1, np.nan, -.1, .2]})
        if duplicate:
            ic.loc[3, 'date'] = self.dates[0]
        top = pd.DataFrame({'prefix': ['Top', None, None, None],
                            'factor_name': ['best', None, None, None],
                            'benchmark': ['univ', None, None, None],
                            'strategy': ['Top_50', None, None, None], 'suffix': ['lag0', None, None, None],
                            'topN': [50, None, None, None], 'trade_date': self.dates,
                            'pf': [0, .1, .21, .089], 'bm': [0, 0, 0, 0], 'excess': [0, .1, .2, .1]})
        with pd.ExcelWriter(path) as writer:
            ic.to_excel(writer, sheet_name='factor@ic_curve', index=False)
            top.to_excel(writer, sheet_name='t50@perf_curve', index=False)

    def test_identity_fill_does_not_fill_ic(self):
        loaded = load_inputs(self.paths, CompareConfig(trade_dates=tuple(self.dates)))
        self.assertTrue(np.isnan(loaded[0].ic.loc[20240103]))
        self.assertEqual(len(loaded[0].top), 4)

    def test_bad_keys_and_missing_selection(self):
        self.write_report(self.paths[0], duplicate=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_inputs(self.paths, CompareConfig(trade_dates=tuple(self.dates)))
        self.write_report(self.paths[0])
        with self.assertRaisesRegex(ValueError, 'available'):
            load_inputs(self.paths, CompareConfig(ic_benchmark='csi300'))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            load_inputs([CompareModelSpec(self.paths[0], name='x'), CompareModelSpec(self.paths[0], name='y')], CompareConfig())

    def test_pipeline_offline_export_and_no_inference(self):
        with patch('src.res.model.analytic.compare.outputs.ArchivedOutputProvider.get', side_effect=AssertionError('inference forbidden')):
            result = compare_models(self.paths, config=CompareConfig(trade_dates=tuple(self.dates)),
                                    output_dir=self.root / 'out', display=False)
        self.assertEqual(len(result.summary), 2)
        self.assertEqual(result.summary.ic_n.tolist(), [3, 3])
        self.assertAlmostEqual(result.summary.top_pf_return.iloc[0], .089)
        self.assertTrue(result.output_paths['pdf'].is_file())
        parameters = json.loads(result.output_paths['parameters'].read_text(encoding='utf-8'))
        self.assertEqual(parameters['config']['trade_dates'], list(self.dates))
        self.assertEqual(parameters['sources'], result.metadata['sources'])
        self.assertEqual(parameters['ic_window'], result.metadata['ic_window'])
        self.assertEqual(parameters['top_window'], result.metadata['top_window'])
        self.assertEqual(parameters['output_dir'], str((self.root / 'out').resolve()))
        self.assertIn('exported_at', parameters)
        with pd.ExcelFile(result.output_paths['xlsx']) as book:
            self.assertIn('Summary', book.sheet_names)
            self.assertIn('Sheet_index', book.sheet_names)

    def test_explicit_excel_does_not_guess_archives(self):
        loaded = load_inputs(self.paths, CompareConfig(templates=('ic',)))
        self.assertIsNone(loaded[0].inference_dir)


class CompareOutputsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for num in (0, 1):
            for date in (20230101, 20240103):
                (self.root / f'archive/{num}/{date}/best').mkdir(parents=True)
        self.model = ModelInput('a', self.root / 'a.xlsx', self.root, 1)

    def test_provider_averages_replicas_and_shares_forward(self):
        provider = ArchivedOutputProvider(self.model, CompareConfig())
        calls = []

        def values(num, checkpoint, date, kinds):
            calls.append((num, checkpoint, date, kinds))
            result = {'pred': pd.DataFrame({'pred.0': [1 + num, 2 + num, 3 + num]}, index=[1, 2, 3])}
            if 'hidden' in kinds:
                result['hidden'] = result['pred'].rename(columns={'pred.0': 'hidden.0'})
            return result

        provider._values = values
        provider._source = lambda num: SimpleNamespace(config=SimpleNamespace(model_param=[{}, {}]))
        score, hidden, checkpoints, errors = provider.get(20240103, True, True)
        np.testing.assert_allclose(score, [1.5, 2.5, 3.5])
        np.testing.assert_allclose(hidden['hidden.0'], [2, 3, 4])
        self.assertEqual(checkpoints, {0: 20230101, 1: 20230101})
        self.assertEqual(calls[1][3], ['pred', 'hidden'])
        self.assertFalse(errors)

    def test_cache_hit_does_not_infer_or_rewrite(self):
        provider = ArchivedOutputProvider(self.model, CompareConfig())
        folder = self.root / 'snapshot/pred_values'
        folder.mkdir(parents=True)
        path = folder / '0.20230101.best.feather'
        pd.DataFrame({'date': [20240102] * 3, 'secid': [3, 1, 2], 'pred.0': [1., 2., 3.]}).to_feather(path)
        original = path.read_bytes()
        provider._source = Mock(side_effect=AssertionError('should not infer'))
        values = provider._values(0, 20230101, 20240102, ['pred'])
        self.assertEqual(len(values['pred']), 3)
        self.assertEqual(path.read_bytes(), original)

    def test_average_snapshot_survives_hidden_failure(self):
        folder = self.root / 'snapshot/pred_recorder/avg_preds'
        folder.mkdir(parents=True)
        path = folder / '20230101.20240102.20240102.feather'
        pd.DataFrame({'date': [20240102] * 3, 'secid': [1, 2, 3], 'submodel': ['best'] * 3,
                      'pred': [1., 2., 3.]}).to_feather(path)
        original = path.read_bytes()
        provider = ArchivedOutputProvider(self.model, CompareConfig())
        provider._values = Mock(side_effect=ValueError('hidden unavailable'))
        score, hidden, _, errors = provider.get(20240102, True, True)
        np.testing.assert_allclose(score, [1, 2, 3])
        self.assertIsNone(hidden)
        self.assertNotIn('pred', errors)
        self.assertIn('hidden', errors)
        self.assertEqual(path.read_bytes(), original)

    def test_two_cache_misses_share_actual_iterator(self):
        provider = ArchivedOutputProvider(self.model, CompareConfig())
        output = pd.DataFrame({'date': [20240102] * 3, 'secid': [1, 2, 3], 'pred.0': [1., 2., 3.]})
        hidden = output.rename(columns={'pred.0': 'hidden.0'})
        batch = SimpleNamespace(batch_date=20240102, output=SimpleNamespace(empty=False, other={'hidden': True}),
                                pred_df=lambda **kw: output, hidden_df=lambda: hidden)
        source = SimpleNamespace(iter_batch_data=Mock(return_value=iter([batch])),
                                 data=SimpleNamespace(early_test_dates=[], storage=SimpleNamespace(del_group=Mock())))
        provider._source = lambda num: source
        values = provider._values(0, 20230101, 20240102, ['pred', 'hidden'])
        self.assertEqual(set(values), {'pred', 'hidden'})
        source.iter_batch_data.assert_called_once()
        values = provider._values(0, 20230101, 20240102, ['pred', 'hidden'])
        source.iter_batch_data.assert_called_once()

    def test_independent_limits_pair_alignment_and_failure_no_replacement(self):
        models = [self.model, ModelInput('b', self.root / 'b.xlsx', self.root, 0)]
        calls = []

        class Provider:
            def __init__(self, model, config):
                self.name = model.name

            def available_dates(self, kind, dates):
                return dates

            def get(self, date, pred, hidden):
                calls.append((self.name, date, pred, hidden))
                if self.name == 'b' and date == 5:
                    return None, None, {}, {'pred': 'test failure', 'hidden': 'test failure'}
                score = pd.Series([1., 2., 3.], index=[3, 1, 2])
                frame = pd.DataFrame({'hidden.0': score, 'hidden.1': score ** 2})
                return score if pred else None, frame if hidden else None, {0: 1}, {}

        config = CompareConfig(analyze_pred=True, analyze_hidden=True, sample_num_corr=3, sample_num_hidden=2)
        tables, matrices, audit, diagnostics = analyze_outputs(models, config, list(range(1, 10)), Provider)
        self.assertEqual(sorted(set(c[1] for c in calls)), [1, 5, 9])
        self.assertEqual(audit.query("kind == 'hidden' and model == '*'").selected_count.iloc[0], 2)
        self.assertEqual(matrices['pred_pearson_count'].loc['a', 'b'], 2)
        self.assertAlmostEqual(matrices['hidden_cka_mean'].loc['a', 'b'], 1)
        self.assertTrue(diagnostics)
        self.assertEqual(set(tables['hidden_dimensions'].date), {1, 9})
        from src.res.model.analytic.compare.report import build_figures
        from dataclasses import asdict
        result = CompareResult(pd.DataFrame({'model': ['a', 'b']}), tables=tables, correlations=matrices,
                               samples=audit, diagnostics=pd.DataFrame(diagnostics),
                               metadata={'config': asdict(config), 'sources': []})
        build_figures(result)
        result.export(self.root / 'optional_report')
        with pd.ExcelFile(result.output_paths['xlsx']) as book:
            self.assertIn('hidden_1', book.sheet_names)
            self.assertIn('pred_pearson', book.sheet_names)
            self.assertIn('hidden_cka', book.sheet_names)


if __name__ == '__main__':
    unittest.main()
