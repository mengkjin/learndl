"""Frozen baseline parity and fail-closed original-data experiment."""
import ast
import importlib.util
import json
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import TestCase, main
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import DataBlock
from src.data.util.classes import data_block
from src.data.preprocess.processors.original_15m import (
    BASELINE_COMMIT, PrePro_15m_original, trade_min_reform, trade_min_fillna,
)
from src.data.preprocess.processors.trade import PrePro_15m
from src.data.preprocess import PrePros
from src.res.model.util.data.prenorm import SingleDataPrenorm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('original_task', ROOT / 'scripts/.autofix/create_15m_original.py')
assert spec is not None and spec.loader is not None
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def block(start=1, end=40):
    return DataBlock(torch.ones(2, end - start + 1, 16, 6), [1, 2], np.arange(start, end + 1), task.FEATURES)


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
        self.assertNotEqual(task.audit(actual)['sha256'], task.audit(current)['sha256'])

    def test_registration_config_and_cache_isolation(self):
        self.assertIn('15m_original', PrePros.keys())
        self.assertNotIn('15m_original', PrePros.enabled_keys())
        self.assertFalse(DataBlock.guess_fillna('15m_original'))
        self.assertFalse(PrePros.use_direct_ffill('15m_original'))
        self.assertFalse(bool(SingleDataPrenorm.from_input('15m_original')))
        self.assertNotEqual(DataBlock.path_preprocess('15m_original', 'fit'), DataBlock.path_preprocess('15m', 'fit'))
        old = yaml.safe_load(subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:configs/schedule/current/gru_15m_new_rtn.yaml'], cwd=ROOT, text=True))
        actual = yaml.safe_load((ROOT / 'configs/schedule/current/gru_15m_original_rtn.yaml').read_text().replace('15m_original', '15m'))
        self.assertEqual(old, actual)

    def test_runpy_main_entrypoint_actually_invokes_task(self):
        import runpy
        task_call = Mock()
        def script_tool(*args, **kwargs):
            def decorate(func):
                def invoke(*args, **kwargs):
                    # Exercise real module entrypoint and main body without DB writes,
                    # task locks or real result email from the ScriptTool wrapper.
                    func.__globals__['run_task'] = task_call
                    return func(*args, **kwargs)
                return invoke
            return decorate
        with patch('src.proj.util.script.ScriptTool', script_tool):
            runpy.run_path(str(ROOT / 'scripts/.autofix/create_15m_original.py'), run_name='__main__')
        task_call.assert_called_once_with(dry_run=False, snapshot_manifest='')

    def test_fresh_script_import_does_not_load_model_or_boost_packages(self):
        import os
        import sys
        code = """
import importlib.abc, importlib.util, sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'src.res.model' or fullname.split('.')[0] in {'lightgbm', 'xgboost', 'catboost', 'optuna'}:
            raise ImportError('unexpected model import: ' + fullname)
sys.meta_path.insert(0, RejectModels())
spec = importlib.util.spec_from_file_location('isolated_autofix', 'scripts/.autofix/create_15m_original.py')
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.REQUEST_ID.endswith('_02')
"""
        with tempfile.TemporaryDirectory() as cache:
            subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                capture_output=True, text=True, timeout=90, env=dict(os.environ, MPLCONFIGDIR=cache))

    def test_audit_window_matches_training_operator_at_ten_percent_boundary(self):
        from typing import cast
        from src.res.model.util.config import ModelConfig
        from src.res.model.util.data.operations import DataOperator
        from src.res.model.util.data.batch_input_loader import DataloaderParam
        b = block()
        b.values[0, 1:4] = torch.nan  # 288 / 2880: exactly 10 percent
        b.values[1, 1:4] = torch.inf
        b.values[1, 4, 0, 0] = -torch.inf  # just over 10 percent
        config = SimpleNamespace(module_type='nn', seq_steps={task.KEY: 1}, input_verify_x_max_nan_ratio=.1)
        op = DataOperator(cast(ModelConfig, config), DataloaderParam(seqlens={task.KEY: 30}))
        values = b.values.clone()
        finite = torch.isfinite(values)
        values[~finite] = torch.nan
        DataBlock.autofill_values(values, b.feature, stretch=True)
        expected = op.finite_position(task.KEY, values, torch.arange(29, 40), finite_mask=finite).sum(0).tolist()
        self.assertEqual(task.input_eligibility(b)['counts'], expected)
        self.assertEqual(expected[0], 1)

    def test_short_history_has_no_30_day_windows(self):
        self.assertEqual(task.input_eligibility(block(1, 5))["counts"], [])

    def test_input_eligibility_rejects_unfilled_and_excess_holes(self):
        b = block()
        b.values[0, :, :, 4] = torch.inf  # >10% holes
        b.values[1, 10, 0, 4] = torch.inf  # forward-fillable
        before = b.values.clone()
        self.assertEqual(task.input_eligibility(b)['counts'], [1] * 11)
        torch.testing.assert_close(b.values, before)
        b.values[1, 0, 0, 4] = torch.nan  # cannot backfill leading price
        self.assertEqual(task.input_eligibility(b)['counts'][0], 0)

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

    def test_snapshot_overlay_recomputes_only_missing_dates(self):
        proc = object.__new__(PrePro_15m_original)
        proc.snapshot = block(2, 3)
        proc.snapshot.values.fill_(7)
        from src.data.preprocess.core import TradePreProcessor
        with patch('src.data.preprocess.processors.original_15m.CALENDAR.range', return_value=np.arange(1, 6)), \
             patch.object(TradePreProcessor, 'pre_process', side_effect=lambda a, b, **kw: block(a, b)) as compute:
            actual = proc.pre_process(1, 5)
        self.assertEqual([call.args for call in compute.call_args_list], [(1, 1), (4, 5)])
        self.assertTrue((actual.values[:, 1:3] == 7).all())
        np.testing.assert_array_equal(actual.date, np.arange(1, 6))

    def test_network_files_are_identical_to_baseline(self):
        for file in ('src/res/algo/nn/model/CNN.py', 'src/res/algo/nn/model/resnet_gru.py',
                     'configs/algo/nn/resnet_gru.yaml'):
            expected = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:{file}'], cwd=ROOT)
            self.assertEqual((ROOT / file).read_bytes(), expected)

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
        config = SimpleNamespace(module_type='nn', seq_steps={task.KEY: 1},
            input_verify_x_max_nan_ratio=.1, input_data_types=[task.KEY], input_final_types=[])
        op = DataOperator(cast(ModelConfig, config), DataloaderParam(seqlens={task.KEY: 30}))
        indices = torch.arange(29, 40)
        labels = torch.ones(2, 40, 1)
        selected = op.effective_samples({task.KEY: values}, labels, indices, {task.KEY: finite})
        assert selected is not None
        for stock, column in selected.nonzero().tolist():
            self.assertEqual(stock, 1)
            end = int(indices[column])
            self.assertTrue(torch.isfinite(values[stock, end - 29:end + 1]).all())
        self.assertEqual(int(selected.sum()), 11)

    def test_leading_missing_history_is_retained_but_internal_gap_fails(self):
        b = block()
        b.values[:, :3] = torch.nan
        self.assertEqual(task.audit(b)['leading_empty_dates'], [1, 2, 3])
        b.values[:, 10] = torch.nan
        with self.assertRaisesRegex(ValueError, 'internal/trailing'):
            task.audit(b)

    def test_preflight_missing_middle_date_fails_without_shortening(self):
        proc = SimpleNamespace(load_start=1, load_end=40, source_start=None, snapshot=None)
        with patch.object(task, 'PrePro_15m_original', return_value=proc), \
             patch.object(task, 'find_snapshot', return_value=(None, {})), \
             patch.object(task.CALENDAR, 'updated', return_value=40), \
             patch.object(task.CALENDAR, 'range', side_effect=lambda a,b: np.arange(a,b+1)), \
             patch.object(task.DB, 'dates', return_value=[d for d in range(4,41) if d != 20]):
            with self.assertRaisesRegex(ValueError, 'coverage incomplete'):
                task.prepare()
        self.assertEqual(proc.load_start, 1)

    def test_result_comparison_aligns_actual_dates_and_marks_missing(self):
        from src.res.model.util.original_comparison import compare_runs
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


class OriginalAutofix(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / 'blocks').mkdir()
        self.stack.enter_context(patch.object(data_block, 'PATH', SimpleNamespace(datablock=self.root / 'blocks')))
        self.stack.enter_context(patch.object(task, 'PATH', SimpleNamespace(interim=self.root, main=ROOT)))
        self.stack.enter_context(patch.object(task.CALENDAR, 'range', side_effect=lambda a, b: np.arange(a, b + 1)))
        self.stack.enter_context(patch.object(task.Logger, 'note'))
        self.request = self.stack.enter_context(patch.object(task, 'request_training', return_value={'id': 'test'}))
        self.procs = []
        self.jobs = []
        for frame in task.FRAMES:
            proc = Mock(frame=frame, load_start=1, load_end=40)
            proc.load_with_extension.side_effect = lambda **kw: block()
            proc.save_dump.side_effect = lambda b, f=frame: b.set_flags(category='preprocess', preprocess_key=task.KEY, frame=f).save_dump()
            proc.load_dump.side_effect = lambda f=frame: DataBlock.load_dump(category='preprocess', preprocess_key=task.KEY, frame=f)
            self.jobs.append((proc, {'frame': frame}))
            self.procs.append(proc)
        self.stack.enter_context(patch.object(task, 'prepare', return_value=self.jobs))

    def test_both_saved_and_verified_before_request(self):
        def requested():
            for proc in self.procs:
                proc.save_dump.assert_called_once()
                proc.load_dump.assert_called_once()
            return {'id': 'test'}
        self.request.side_effect = requested
        report = task.run_task()
        self.assertEqual(report['status'], 'training_requested')
        self.assertEqual([item['status'] for item in report['blocks']], ['verified', 'verified'])

    def test_failure_does_not_request_training(self):
        self.procs[1].load_dump.return_value = DataBlock()
        self.procs[1].load_dump.side_effect = None
        with self.assertRaisesRegex(ValueError, 'schema'):
            task.run_task()
        self.request.assert_not_called()
        report = json.loads(next(self.root.rglob('report.json')).read_text())
        self.assertEqual(report['status'], 'failed')

    def test_dry_run_no_writes_or_request(self):
        task.run_task(dry_run=True)
        self.assertEqual([p.name for p in self.root.iterdir()], ['blocks'])
        self.request.assert_not_called()

    def test_unverified_old_report_is_not_trusted(self):
        folder = self.root / 'autofix_backups' / 'vwap_close_old'
        folder.mkdir(parents=True)
        (folder / 'report.json').write_text(json.dumps({'blocks': [{'key': '15m', 'frame': 'fit'}]}))
        snapshot, evidence = task.find_snapshot('fit')
        self.assertIsNone(snapshot)
        self.assertEqual(evidence['kind'], 'legacy_formula_recomputed')
        with self.assertRaisesRegex(ValueError, 'attest'):
            task.find_snapshot('fit', str(folder / 'report.json'))

    def test_verified_snapshot_requires_content_hash(self):
        b = block()
        b.set_flags(category='preprocess', preprocess_key='15m', frame='fit').save_dump()
        path = DataBlock.path_preprocess('15m', 'fit')
        manifest = self.root / 'snapshot.json'
        doc = {'baseline_commit': BASELINE_COMMIT, 'historical_snapshot': True, 'blocks': [
            {'key': '15m', 'frame': 'fit', 'backup': str(path), 'sha256': task.audit(b)['sha256']}]}
        manifest.write_text(json.dumps(doc))
        actual, _ = task.find_snapshot('fit', str(manifest))
        torch.testing.assert_close(actual.values, b.values)
        doc['blocks'][0]['sha256'] = 'bad'
        manifest.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'digest'):
            task.find_snapshot('fit', str(manifest))


class OriginalRealPipeline(TestCase):
    def test_real_chunks_preprocessing_save_audit_then_training_request(self):
        from src.data.loader import BlockLoader
        dates = task.CALENDAR.range(20151102, 20160219)
        factory = PrePro_15m_original
        def processor(frame, **kwargs):
            proc = factory(frame, **kwargs)
            proc.load_start = 20151102 if frame == 'fit' else 20160201
            proc.mask = {}
            return proc
        minute = pd.DataFrame({'secid': np.repeat([1, 2], 240), 'minute': np.tile(np.arange(240), 2),
            **{key: np.ones(480, dtype=np.float32) for key in
                ('open', 'close', 'high', 'low', 'vwap', 'amount', 'volume')}})
        minute.loc[minute.minute < 15, 'volume'] = 0
        def day(source, key, start, end, feature=None, **kwargs):
            selected = dates[(dates >= start) & (dates <= end)]
            return DataBlock(torch.ones(2, len(selected), 1, 2), [1, 2], selected,
                             ['turn_fl', 'preclose']).align_feature(feature)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / 'blocks').mkdir()
            (root / 'raw' / 'day').mkdir(parents=True)
            stack.enter_context(patch.object(task, 'PATH', SimpleNamespace(interim=root, main=ROOT)))
            stack.enter_context(patch.object(data_block, 'PATH', SimpleNamespace(datablock=root / 'blocks')))
            stack.enter_context(patch.object(task.CALENDAR, 'updated', return_value=20160219))
            stack.enter_context(patch.object(task.DB, 'dates', return_value=dates))
            stack.enter_context(patch.object(task.DB, 'load', return_value=minute))
            stack.enter_context(patch.object(task, 'PrePro_15m_original', side_effect=processor))
            stack.enter_context(patch.object(BlockLoader, 'src_path', new_callable=PropertyMock, return_value=root / 'raw'))
            stack.enter_context(patch.object(DataBlock, 'load_raw', side_effect=day))
            stack.enter_context(patch.object(task.Logger, 'note'))
            request = stack.enter_context(patch.object(task, 'request_training', return_value={'id': 'test'}))
            report = task.run_task()
            request.assert_called_once()
            self.assertEqual(report['status'], 'training_requested')
            for item in report['blocks']:
                self.assertEqual(item['status'], 'verified')
                self.assertTrue(item['minute_sources'])
                self.assertGreater(sum(sum(row['inf']) for row in item['audit']['daily']), 0)

    def test_autofix_manifest(self):
        from src.api.task_monitor.scheduling.autofix_spec import manifest, validate
        entries = manifest(ROOT / 'runs/autofix/tasks.yaml')
        matching = [item for item in entries if item['id'] == task.REQUEST_ID]
        self.assertEqual(len(matching), 1)
        validate(ROOT, matching[0])
        self.assertTrue(matching[0]['enabled'])
        previous = next(item for item in entries if item['id'] == '20261021_create_15m_original_train_01')
        self.assertFalse(previous['enabled'])
        self.assertEqual(sum(item.get('enabled', True) for item in entries
            if item['script'] == matching[0]['script']), 1)


if __name__ == '__main__':
    main()
