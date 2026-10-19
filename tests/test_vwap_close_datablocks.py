"""VWAP-only repair, rolling-week alignment, and full Autofix rebuild safety."""
import importlib.util
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.data import DataBlock
from src.data.preprocess.core import TradePreProcessor
from src.data.preprocess.processors.trade import (
    PrePro_15m, PrePro_30m, PrePro_60m, PrePro_30mcont, PrePro_week,
)
from src.data.util.classes import data_block

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_vwap_close', ROOT / 'scripts/.autofix/repair_vwap_close_datablocks.py')
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)
PROCESSORS = {'15m': PrePro_15m, '30m': PrePro_30m, '60m': PrePro_60m,
              '30mcont': PrePro_30mcont, 'week': PrePro_week}


def source_blocks(key):
    dates = np.arange(20240101, 20240109)
    if key == 'week':
        features = ['adjfactor', 'preclose', *task.FEATURES]
        values = torch.ones(2, 8, 1, 8, dtype=torch.float64)
        values[..., 0] = 2.
        values[..., 1] = 10.
        values[..., 2:7] = 11.
        values[:, :, 0, 3] = torch.arange(12., 20.)  # each day's close differs
        return {'day': DataBlock(values, [1, 2], dates, features)}
    features = ['close', 'high', 'low', 'open', 'volume', 'vwap']
    values = torch.full((2, 8, task.BARS[key], 6), 11., dtype=torch.float64)
    values[..., 0] = 12.
    values[..., 4] = 100.
    day = torch.tensor([10., 1.], dtype=torch.float64).reshape(1, 1, 1, 2).repeat(2, 8, 1, 1)
    return {key if key != '30mcont' else '30m': DataBlock(values, [1, 2], dates, features),
            'day': DataBlock(day, [1, 2], dates, ['preclose', 'turn_fl'])}


def clone_blocks(blocks):
    return {key: DataBlock(block.values.clone(), block.secid.copy(), block.date.copy(), block.feature.copy())
            for key, block in blocks.items()}


class VwapCloseTest(TestCase):
    def test_all_five_processors_match_same_source_bar_close_replacement(self):
        for key, cls in PROCESSORS.items():
            with self.subTest(key=key):
                blocks = source_blocks(key)
                raw = next(iter(blocks.values()))
                v, c = [list(raw.feature).index(f) for f in ('vwap', 'close')]
                raw.values[0, 3, 0, v] = torch.inf
                raw.values[1, 4, 0, v] = -torch.inf
                expected_inputs = clone_blocks(blocks)
                expected_raw = next(iter(expected_inputs.values()))
                mask = torch.isinf(expected_raw.values[..., v])
                expected_raw.values[..., v][mask] = expected_raw.values[..., c][mask]
                proc = object.__new__(cls)
                expected = TradePreProcessor.process_blocks(proc, expected_inputs)
                records = []
                proc.vwap_inf_repair_observer = records.append
                actual = proc.process_blocks(blocks)
                torch.testing.assert_close(actual.values, expected.values, equal_nan=True)
                self.assertFalse(torch.isinf(actual.values[..., 4]).any())
                self.assertEqual(sum(r['inf_count'] for r in records), 2)
                if key == 'week':
                    # Day 4 appears at different inday positions in later windows.
                    for end_day in range(4, 8):
                        bar = 4 - (end_day - 4)
                        self.assertEqual(actual.values[1, end_day, bar, 4], actual.values[1, end_day, bar, 1])

    def test_only_infinite_vwap_changes_including_invalid_close(self):
        values = torch.ones(1, 1, 6, 6)
        values[0, 0, :, 4] = torch.tensor([torch.inf, -torch.inf, torch.nan, 7., torch.inf, -torch.inf])
        values[0, 0, :, 1] = torch.tensor([2., 3., 4., 5., torch.nan, torch.inf])
        values[..., 2] = torch.inf  # Other fields must not be sanitized.
        block = DataBlock(values.clone(), [1], [20240102], task.FEATURES)
        object.__new__(PrePro_15m)._repair_vwap_inf(block, 'test')
        expected = values.clone()
        expected[0, 0, :, 4] = torch.tensor([2., 3., torch.nan, 7., torch.nan, torch.nan])
        torch.testing.assert_close(block.values, expected, equal_nan=True)

    def test_float32_overflow_is_repaired_after_normalisation(self):
        for key, cls in PROCESSORS.items():
            with self.subTest(key=key):
                blocks = source_blocks(key)
                raw = next(iter(blocks.values()))
                raw.values[0, 4, 0, list(raw.feature).index('vwap')] = 1e40
                result = object.__new__(cls).process_blocks(blocks)
                bar = 4 if key == 'week' else 0
                self.assertEqual(result.values[0, 4, bar, 4], result.values[0, 4, bar, 1])
                self.assertFalse(torch.isinf(result.values[..., 4]).any())


class VwapAutofixTest(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / 'blocks').mkdir()
        self.stack.enter_context(patch.object(data_block, 'PATH', SimpleNamespace(datablock=self.root / 'blocks')))
        self.stack.enter_context(patch.object(task, 'PATH', SimpleNamespace(interim=self.root)))
        self.stack.enter_context(patch.object(task.CALENDAR, 'updated', return_value=30))
        self.stack.enter_context(patch.object(task.CALENDAR, 'range', side_effect=lambda start, end: np.arange(start, end + 1)))
        self.stack.enter_context(patch.object(task.CALENDAR, 'td', side_effect=lambda date, offset: SimpleNamespace(td=date + offset)))
        self.dates = self.stack.enter_context(patch.object(task.DB, 'dates', return_value=np.arange(1, 31)))
        self.stack.enter_context(patch.object(task.Logger, 'note'))
        self.stack.enter_context(patch.object(task.PreProHistNorm, 'norm_path', side_effect=lambda key: self.root / f'{key}.norm'))
        self.norms = self.stack.enter_context(patch.object(task.PreProHistNorm, 'load_keys', side_effect=lambda keys: {
            keys[0]: SimpleNamespace(avg=torch.ones(1), std=torch.ones(1))}))
        self.procs = {}
        for frame in task.FRAMES:
            for key in task.SOURCES:
                proc = Mock(frame=frame, load_start=5 if frame == 'fit' else 20)
                proc.load_with_extension.side_effect = lambda p=proc, k=key, **kw: self.block(k, p.load_start)
                proc.save_dump.side_effect = lambda block, k=key, f=frame: self.write(block, k, f)
                proc.load_dump.side_effect = lambda k=key, f=frame: DataBlock.load_dump(category='preprocess', preprocess_key=k, frame=f)
                self.procs[key, frame] = proc
                self.write(self.block(key, proc.load_start), key, frame)
        self.stack.enter_context(patch.object(task.PrePros, 'get_processor', side_effect=lambda key, frame, **kw: self.procs[key, frame]))

    def block(self, key, start):
        dates = np.arange(start, 31)
        return DataBlock(torch.ones(2, len(dates), task.BARS[key], 6), [1, 2], dates, task.FEATURES)

    def write(self, block, key, frame):
        block.set_flags(category='preprocess', preprocess_key=key, frame=frame).save_dump()

    def report(self):
        return json.loads(next(self.root.rglob('report.json')).read_text())

    def test_all_ten_healthy_blocks_are_backed_up_and_rebuilt_with_norms(self):
        for key in task.SOURCES:
            if key in task.HistNormTasks:
                (self.root / f'{key}.norm').write_text('old norm')
        def first_rebuild(**kwargs):
            for item in self.report()['blocks']:
                self.assertTrue(Path(item['backup']).exists())
                if 'norm_backup' in item:
                    self.assertEqual(Path(item['norm_backup']).read_text(), 'old norm')
            return self.block('15m', 5)
        self.procs['15m', 'fit'].load_with_extension.side_effect = first_rebuild
        task.run_task()
        report = self.report()
        self.assertEqual(report['status'], 'success')
        self.assertEqual(len(report['blocks']), 10)
        for item in report['blocks']:
            proc = self.procs[item['key'], item['frame']]
            proc.load_with_extension.assert_called_once_with(reconstruct=True)
            self.assertTrue(Path(item['backup']).exists())
            self.assertEqual(item['status'], 'verified')
            self.assertEqual(proc.save_norm.call_count, int(item['frame'] == 'fit' and item['key'] in task.HistNormTasks))
        self.assertEqual(self.norms.call_count, 4)

    def test_invalid_norm_stops_before_saving_datablock(self):
        self.norms.side_effect = lambda keys: {keys[0]: SimpleNamespace(avg=torch.tensor([torch.inf]), std=torch.ones(1))}
        with self.assertRaisesRegex(ValueError, 'historical norm'):
            task.run_task()
        self.assertEqual(self.report()['status'], 'failed')
        self.procs['15m', 'fit'].save_dump.assert_not_called()
        self.procs['30m', 'fit'].load_with_extension.assert_not_called()

    def test_dry_run_checks_all_sources_without_writes(self):
        self.assertEqual(len(task.run_task(dry_run=True)), 10)
        self.assertFalse((self.root / 'autofix_backups').exists())
        for proc in self.procs.values():
            proc.load_with_extension.assert_not_called()
            proc.save_dump.assert_not_called()

    def test_source_gaps_and_missing_week_warmup_stop_before_any_rebuild(self):
        for absent in (1, 15, 30):
            with self.subTest(absent=absent):
                self.dates.return_value = np.setdiff1d(np.arange(1, 31), [absent])
                # Predict week must keep the full requested window plus warmup.
                self.procs['week', 'predict'].load_start = 5
                with self.assertRaisesRegex(ValueError, 'incomplete source coverage'):
                    task.run_task()
                for proc in self.procs.values():
                    proc.load_with_extension.assert_not_called()

    def test_fit_allows_only_unavailable_leading_history(self):
        proc = self.procs['15m', 'fit']
        proc.load_end = 30
        item = {'key': '15m'}
        task.check_sources(proc, item, {'15min': np.arange(10, 31), 'day': np.arange(1, 31)})
        self.assertEqual(proc.load_start, 10)
        self.assertEqual(item['source_coverage']['omitted_leading_dates'], list(range(5, 10)))

    def test_bad_rebuild_and_same_shape_saved_corruption_are_detected(self):
        for failure in ('dates', 'vwap', 'other', 'saved'):
            with self.subTest(failure=failure):
                proc = self.procs['15m', 'fit']
                bad = self.block('15m', 5)
                if failure == 'dates':
                    bad = bad.slice_date(6, None)
                elif failure in ('vwap', 'other'):
                    bad.values[0, 0, 0, 4 if failure == 'vwap' else 2] = torch.inf
                proc.load_with_extension.side_effect = lambda **kw: bad
                if failure == 'saved':
                    def corrupt(block):
                        block.values[0, 0, 0, 4] += 1.
                        self.write(block, '15m', 'fit')
                    proc.save_dump.side_effect = corrupt
                with self.assertRaises(ValueError):
                    task.run_task()
                self.procs['30m', 'fit'].load_with_extension.assert_not_called()

    def test_manifest_is_valid_and_task_id_is_unique(self):
        from src.api.task_monitor.scheduling.autofix_spec import manifest, validate
        entries = manifest(ROOT / 'runs/autofix/tasks.yaml')
        item = next(item for item in entries if item['id'] == '20261018_repair_vwap_close_datablocks_05')
        validate(ROOT, item)
        self.assertTrue(item['enabled'])


if __name__ == '__main__':
    main()
