"""Frozen baseline parity and fail-closed original-data experiment."""
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import DataBlock
from src.data.preprocess.processors.original_15m import (
    BASELINE_COMMIT, PrePro_15m_original, trade_min_reform,
)
from src.data.preprocess.processors.trade import PrePro_15m
from src.data.preprocess import PrePros
from src.res.model.util.data.prenorm import SingleDataPrenorm

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'turn_fl']


ORIGINAL_KEY = '15m_original'


def block(start=1, end=40):
    return DataBlock(torch.ones(2, end - start + 1, 16, 6), [1, 2], np.arange(start, end + 1), ORIGINAL_FEATURES)


def legacy_functions():
    text = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:src/data/util/transform.py'], cwd=ROOT, text=True)
    parsed = ast.parse(text)
    funcs: list[ast.stmt] = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
             and node.name in ('trade_min_fillna', 'trade_min_reform')]
    namespace: dict[str, Any] = {'pd': pd, 'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<dd51-reference>', 'exec'), namespace)
    return namespace


class OriginalParity(TestCase):
    def test_frozen_aggregation_matches_git_baseline_edge_cases(self):
        old = legacy_functions()
        for dtype in (np.float32, np.float64):
            for case in ('normal', 'zero', 'positive_over_zero', 'inf', 'nan', 'overflow', 'missing'):
                with self.subTest(dtype=dtype, case=case):
                    df = pd.DataFrame({'secid': 1, 'minute': np.arange(240),
                        **{key: np.ones(240, dtype=dtype) for key in
                           ('open', 'close', 'high', 'low', 'vwap', 'amount', 'volume')}})
                    if case in ('zero', 'positive_over_zero'):
                        df.loc[:14, 'volume'] = 0
                        if case == 'zero':
                            df.loc[:14, 'amount'] = 0
                    if case in ('inf', 'nan', 'overflow'):
                        df.loc[:14, 'amount'] = {'inf': np.inf, 'nan': np.nan, 'overflow': 1e38}[case]
                    if case == 'missing':
                        df = df.drop([2, 5, 8])
                        df.loc[20, 'close'] = np.nan
                    with np.errstate(all='ignore'):
                        actual = trade_min_reform(df.copy(), 15)
                        expected = old['trade_min_reform'](df.copy(), 15)
                    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_preprocessor_matches_dd51_and_preserves_inf(self):
        from src.data.preprocess.core import TradePreProcessor
        text = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:src/data/preprocess/processors/trade.py'], cwd=ROOT, text=True)
        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'PrePro_15m')
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process')
        scope = {}
        exec(compile(ast.Module(body=[process], type_ignores=[]), '<dd51-process>', 'exec'), scope)
        class Reference(TradePreProcessor):
            def block_loaders(self, **kwargs):
                return {}
            process = scope['process']
        def sources():
            v = torch.ones(2, 40, 16, 6, dtype=torch.float64)
            v[..., 0] = 10
            v[0, 3, :4, 5] = torch.tensor([torch.inf, -torch.inf, torch.nan, 1e40], dtype=torch.float64)
            return {'15m': DataBlock(v, [1, 2], np.arange(1, 41), ['close', 'high', 'low', 'open', 'volume', 'vwap']),
                'day': DataBlock(torch.ones(2, 40, 1, 2), [1, 2], np.arange(1, 41), ['preclose', 'turn_fl'])}
        expected = object.__new__(Reference).process_blocks(sources())
        actual = object.__new__(PrePro_15m_original).process_blocks(sources())
        torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(int(torch.isinf(actual.values).sum()), 3)
        current = object.__new__(PrePro_15m).process_blocks(sources())
        self.assertFalse(torch.isinf(current.values).any())

    def test_registration_config_and_cache_isolation(self):
        self.assertIn('15m_original', PrePros.keys())
        self.assertNotIn('15m_original', PrePros.enabled_keys())
        self.assertFalse(DataBlock.guess_fillna('15m_original'))
        self.assertFalse(PrePros.use_direct_ffill('15m_original'))
        self.assertFalse(bool(SingleDataPrenorm.from_input('15m_original')))
        self.assertNotEqual(DataBlock.path_preprocess('15m_original', 'fit'), DataBlock.path_preprocess('15m', 'fit'))
        old = yaml.safe_load(subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:configs/schedule/current/gru_15m_new_rtn.yaml'], cwd=ROOT, text=True))
        actual = yaml.safe_load((ROOT / 'configs/schedule/archive/gru_15m_original_rtn.yaml').read_text().replace('15m_original', '15m'))
        self.assertEqual(actual['algo']['resnet_gru'].pop('resnet_pool'), ['flatten'])
        self.assertEqual(old, actual)

    def test_source_loader_uses_minute_data_not_fixed_15min(self):
        proc = object.__new__(PrePro_15m_original)
        raw = pd.DataFrame({'secid': 1, 'minute': np.arange(240),
            **{key: np.ones(240, dtype=np.float32) for key in
               ('open', 'close', 'high', 'low', 'vwap', 'amount', 'volume')}})
        raw.loc[:14, 'volume'] = 0
        day = DataBlock(torch.ones(1, 2, 1, 2), [1], [1, 2], ['turn_fl', 'preclose'])
        with patch('src.data.preprocess.processors.original_15m.CALENDAR.range', return_value=np.array([1, 2])), \
             patch('src.data.preprocess.processors.original_15m.DB.load', return_value=raw) as load, \
             patch.object(proc, 'block_loaders', return_value={'day': Mock(load=Mock(return_value=day))}):
            actual = proc.process_blocks(proc.load_blocks(1, 2))
        self.assertEqual(actual.shape, (1, 2, 16, 6))
        self.assertTrue(torch.isinf(actual.values[0, :, 0, 4]).all())
        self.assertEqual([call.args[1] for call in load.call_args_list], ['min', 'min'])

    def test_original_network_loads_baseline_weights_and_reproduces_outputs(self):
        from src.res.algo.nn.model.resnet_gru import resnet_gru

        def baseline_module(name):
            package = 'src.res.algo.nn.model'
            path = f'src/res/algo/nn/model/{name}.py'
            source = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:{path}'], cwd=ROOT)
            module = ModuleType(f'{package}.{name}')
            module.__package__ = package
            exec(compile(source, path, 'exec'), module.__dict__)
            return module

        cnn = baseline_module('CNN')
        with patch.dict(sys.modules, {cnn.__name__: cnn}):
            reference_type = baseline_module('resnet_gru').resnet_gru
        config = yaml.safe_load((ROOT / 'configs/schedule/archive/gru_15m_original_rtn.yaml').read_text())
        params = {key: values[0] for key, values in config['algo']['resnet_gru'].items()}
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(17)
            reference = reference_type(6, inday_dim=16, **params).eval()
            current = resnet_gru(6, inday_dim=16, **params).eval()
            current.load_state_dict(reference.state_dict(), strict=True)
            inputs = torch.randn(3, 30, 16, 6)
            expected, before = reference(inputs)
            actual, after = current(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(after['hidden'], before['hidden'], rtol=0, atol=0)

    def test_real_input_selection_materializes_only_finite_batches(self):
        from src.res.model.util.data.operations import DataOperator
        from src.res.model.util.data.batch_input_loader import DataloaderParam
        from typing import cast
        from src.res.model.util.config import ModelConfig
        b = block()
        b.values[0, :, :, 4] = torch.inf
        b.values[1, 10, 0, 4] = torch.inf
        values = b.values.clone()
        finite = torch.isfinite(values)
        values[~finite] = torch.nan
        DataBlock.autofill_values(values, b.feature, stretch=True)
        config = SimpleNamespace(module_type='nn', seq_steps={ORIGINAL_KEY: 1},
            input_verify_x_max_nan_ratio=.1, input_data_types=[ORIGINAL_KEY], input_final_types=[])
        op = DataOperator(cast(ModelConfig, config), DataloaderParam(seqlens={ORIGINAL_KEY: 30}))
        indices = torch.arange(29, 40)
        labels = torch.ones(2, 40, 1)
        selected = op.effective_samples({ORIGINAL_KEY: values}, labels, indices, {ORIGINAL_KEY: finite})
        assert selected is not None
        for stock, column in selected.nonzero().tolist():
            self.assertEqual(stock, 1)
            end = int(indices[column])
            self.assertTrue(torch.isfinite(values[stock, end - 29:end + 1]).all())
        self.assertEqual(int(selected.sum()), 11)

    def test_loader_skips_missing_minute_files_only_in_relaxed_mode(self):
        proc = object.__new__(PrePro_15m_original)
        proc.source_observer = Mock()
        raw = pd.DataFrame({'secid': 1, 'minute': np.arange(240),
            **{key: np.ones(240, dtype=np.float32) for key in
               ('open', 'close', 'high', 'low', 'vwap', 'amount', 'volume')}})
        day = DataBlock(torch.ones(1, 3, 1, 2), [1], [1,2,3], ['turn_fl', 'preclose'])
        with patch('src.data.preprocess.processors.original_15m.CALENDAR.range', return_value=np.array([1,2,3])), \
             patch('src.data.preprocess.processors.original_15m.DB.load', side_effect=lambda *a, **kw: pd.DataFrame() if a[2] == 2 else raw), \
             patch.object(proc, 'block_loaders', return_value={'day': Mock(load=Mock(return_value=day))}):
            actual = proc.process_blocks(proc.load_blocks(1,3))
            self.assertEqual(actual.date.tolist(), [1,2,3])
            self.assertTrue(torch.isnan(actual.values[:, 1]).all())
            proc.source_observer.assert_called_with({'date': 3, 'minute_rows': 240,
                'amount_dtype': 'float32', 'volume_dtype': 'float32', 'bar_rows': 16})
            self.assertIn({'date': 2, 'status': 'empty_minute_source'},
                          [call.args[0] for call in proc.source_observer.call_args_list])
            proc.require_continuous_dates = True
            with self.assertRaisesRegex(ValueError, 'missing minute source 2'):
                proc.load_blocks(1,3)

    def test_result_comparison_aligns_actual_dates_and_marks_missing(self):
        from src.res.model.analytic.original_comparison import compare_runs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = []
            for name, ic in [('old', .1), ('new', .2)]:
                path = root / (name + '.txt')
                path.write_text(f'model #0 @20170103|Trial1-0 ModelEpoch#42 EarlyStop|BestAttempt Trial0-0 TrainIC 0.3 ValidIC {ic}|Cost')
                logs.append(path)
                folder = root / name / 'snapshot' / 'basic_test'
                folder.mkdir(parents=True)
                pd.DataFrame({'model_num': [0,0], 'model_date': [20170103]*2,
                    'submodel': ['best']*2, 'date': [1,2] if name == 'old' else [2,3],
                    'rankic_rtn': [.1,.2] if name == 'old' else [.5,.8]}).to_feather(folder / 'test_by_date.feather')
            report = compare_runs(logs[0], logs[1], root / 'old', root / 'new')
            self.assertIsNone(report['models'][0]['delta']['effective_samples'])
            self.assertAlmostEqual(report['models'][0]['delta']['valid_ic'], .1)
            metric = report['test'][0]['metrics']['rankic_rtn']
            self.assertEqual(metric['common_dates'], [2])
            self.assertAlmostEqual(metric['delta'], .3)
            self.assertIsNone(report['test'][0]['metrics']['top5pct'])



if __name__ == '__main__':
    main()
