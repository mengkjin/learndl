"""Exercise repair/skip/failure paths using temporary dumps, never server data."""
import importlib.util
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.data import DataBlock
from src.data.util.classes import data_block

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/.autofix/repair_15mraw_datablocks.py'
spec = importlib.util.spec_from_file_location('repair_15mraw', SCRIPT)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def make_block(legacy=False):
    features = task.EXPECTED_FEATURES.copy()
    values = torch.ones(2, 30, 16, 6)
    if legacy:
        features[-1] = 'turn_fl'
        values[..., -1] = torch.nan
    return DataBlock(values, np.array([1, 2]), np.arange(20240101, 20240131), np.array(features))


class Repair15mrawTest(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / 'blocks').mkdir()
        self.stack.enter_context(patch.object(data_block, 'PATH', SimpleNamespace(datablock=self.root / 'blocks')))
        self.stack.enter_context(patch.object(task, 'PATH', SimpleNamespace(interim=self.root)))
        self.stack.enter_context(patch.object(task.CALENDAR, 'updated', return_value=20240130))
        self.stack.enter_context(patch.object(task.CALENDAR, 'range', side_effect=lambda start, end: np.arange(start, end + 1)))
        self.stack.enter_context(patch('src.proj.DB.dates', return_value=np.arange(20240101, 20240131)))
        self.stack.enter_context(patch.object(task.Logger, 'note'))
        self.training_original = task.request_training
        self.training = self.stack.enter_context(patch.object(task, 'request_training'))
        self.procs = {}
        for frame in task.FRAMES:
            proc = Mock()
            proc.frame = frame
            proc.load_start = 20240101
            proc.load_dump.side_effect = lambda f=frame: DataBlock.load_dump(category='preprocess', preprocess_key='15mraw', frame=f)
            proc.load_with_extension.side_effect = lambda **kwargs: make_block()
            proc.save_dump.side_effect = lambda block, f=frame: self.write(block, f)
            self.procs[frame] = proc
        self.factory = self.stack.enter_context(patch.object(task.PrePros, 'get_processor',
            side_effect=lambda key, frame, **kwargs: self.procs[frame]))

    def write(self, block, frame):
        block.set_flags(category='preprocess', preprocess_key='15mraw', frame=frame).save_dump()

    def report(self):
        paths = list(self.root.rglob('report.json'))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_repairs_legacy_dump_backs_it_up_and_skips_healthy_frame_and_rerun(self):
        self.write(make_block(legacy=True), 'fit')
        self.write(make_block(), 'predict')
        task.run_task()
        self.training.assert_called_once()
        self.procs['fit'].load_with_extension.assert_called_once_with(reconstruct=True)
        self.procs['predict'].load_with_extension.assert_not_called()
        self.assertIsNone(task.block_issue(self.procs['fit'].load_dump()))
        report = self.report()
        self.assertEqual(report['status'], 'success')
        fit = report['frames'][0]
        self.assertEqual(fit['status'], 'verified')
        self.assertTrue(Path(fit['backup']).is_dir())
        from src.proj import Load
        backup = Load.mmap(Path(fit['backup']), values=True, index=True)
        self.assertEqual(list(backup['index']['feature'])[-1], 'turn_fl')
        self.assertTrue(torch.isnan(backup['values'][..., -1]).all())
        task.run_task()
        self.procs['fit'].load_with_extension.assert_called_once()
        self.assertEqual(len(list(self.root.rglob('report.json'))), 1)

    def test_missing_frames_are_built_without_fake_backup(self):
        def request_after_validation():
            self.assertEqual(self.report()['status'], 'success')
            for proc in self.procs.values():
                self.assertIsNone(task.block_issue(proc.load_dump()))
        self.training.side_effect = request_after_validation
        task.run_task()
        self.training.assert_called_once()
        report = self.report()
        for item in report['frames']:
            self.assertIsNone(item['backup'])
            self.assertEqual(item['status'], 'verified')

    def test_dry_run_only_inspects(self):
        self.write(make_block(legacy=True), 'fit')
        plan = task.run_task(dry_run=True)
        self.assertTrue(all(item['reason'] for item in plan))
        self.assertFalse((self.root / 'autofix_backups').exists())
        self.training.assert_not_called()
        for proc in self.procs.values():
            proc.load_with_extension.assert_not_called()
            proc.save_dump.assert_not_called()

    def test_invalid_rebuild_stops_before_overwriting_original(self):
        self.write(make_block(legacy=True), 'fit')
        self.procs['fit'].load_with_extension.side_effect = lambda **kwargs: make_block(legacy=True)
        with self.assertRaisesRegex(ValueError, 'rebuilt block.*invalid schema'):
            task.run_task()
        self.procs['fit'].save_dump.assert_not_called()
        self.procs['predict'].load_with_extension.assert_not_called()
        self.assertIn('turn_fl', self.procs['fit'].load_dump().feature)
        self.assertEqual(self.report()['status'], 'failed')
        self.training.assert_not_called()

    def test_failed_saved_validation_stops_and_preserves_backup(self):
        self.write(make_block(legacy=True), 'fit')
        self.procs['fit'].save_dump.side_effect = lambda block: self.write(make_block(legacy=True), 'fit')
        with self.assertRaisesRegex(ValueError, 'saved block.*invalid schema'):
            task.run_task()
        report = self.report()
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(Path(report['frames'][0]['backup']).exists())
        self.procs['predict'].load_with_extension.assert_not_called()
        self.training.assert_not_called()

    def test_incomplete_rebuild_is_not_published_or_trained(self):
        self.write(make_block(legacy=True), 'fit')
        self.procs['fit'].load_with_extension.side_effect = lambda **kwargs: make_block().slice_date(20240102, None)
        with self.assertRaisesRegex(ValueError, 'rebuilt dates do not cover'):
            task.run_task()
        self.procs['fit'].save_dump.assert_not_called()
        self.training.assert_not_called()
        self.assertIn('turn_fl', self.procs['fit'].load_dump().feature)
        self.assertEqual(self.report()['status'], 'failed')

    def test_healthy_frames_still_request_training(self):
        for frame in task.FRAMES:
            self.write(make_block(), frame)
        task.run_task()
        self.training.assert_called_once()
        for proc in self.procs.values():
            proc.load_with_extension.assert_not_called()

    def test_training_request_failure_propagates_after_successful_data_repair(self):
        self.training.side_effect = RuntimeError('training worklist unavailable')
        with self.assertRaisesRegex(RuntimeError, 'training worklist unavailable'):
            task.run_task()
        self.assertEqual(self.report()['status'], 'success')

    def test_request_targets_only_gru_15mraw_with_stable_id(self):
        # Exercise the real helper separately from the mocked GPU handoff above.
        from src.api.calls.worklist_state import WorklistState
        calls = []
        def request(state, request_id):
            calls.append((state.schedule_name, request_id))
            return {'id': request_id}
        with patch.object(WorklistState, 'request_training', request):
            for context in (None, {'id': 'new-autofix-id'}):
                with patch('src.automation.scheduling.maintenance_gate.sandbox_request', return_value=context):
                    self.training_original()
        self.assertEqual(calls, [('gru_15mraw', task.TRAINING_REQUEST_ID), ('gru_15mraw', 'new-autofix-id')])

    def test_all_nan_or_infinite_features_trigger_repair(self):
        for bad in (torch.nan, torch.inf):
            with self.subTest(value=bad):
                block = make_block()
                block.values[..., -1] = bad
                self.assertIsNotNone(task.block_issue(block))

    def test_new_task_traces_then_saves_nan_and_requests_training(self):
        spec = importlib.util.spec_from_file_location('repair_inf_integration', SCRIPT.with_name('repair_15mraw_inf.py'))
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        block = make_block()
        block.values[0, 0, 0, 4] = torch.inf
        self.write(make_block(legacy=True), 'fit')
        self.write(make_block(), 'predict')
        self.procs['fit'].frame = 'fit'
        self.procs['fit'].load_with_extension.side_effect = lambda **kwargs: block
        raw = pd.DataFrame([{'secid': 1, 'minute': 0, 'vwap': np.inf, 'amount': 1., 'volume': 0.}])
        with patch.object(new, 'repair', task), patch.object(new, 'load_source_day', return_value=(raw, '/raw/15min.feather')):
            new.run_task()
        saved = self.procs['fit'].load_dump()
        self.assertFalse(torch.isinf(saved.values).any())
        self.assertTrue(torch.isnan(saved.values[0, 0, 0, 4]))
        self.training.assert_called_once_with(new.REQUEST_ID)
        self.assertEqual(self.report()['frames'][0]['inf_audit']['converted_to_nan'], 1)
        self.assertEqual(self.report()['status'], 'success')

    def test_new_task_unexplained_inf_keeps_original_and_records_failure(self):
        spec = importlib.util.spec_from_file_location('repair_inf_failure', SCRIPT.with_name('repair_15mraw_inf.py'))
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        block = make_block()
        block.values[0, 0, 0, 4] = torch.inf
        self.write(make_block(legacy=True), 'fit')
        self.procs['fit'].frame = 'fit'
        self.procs['fit'].load_with_extension.side_effect = lambda **kwargs: block
        raw = pd.DataFrame([{'secid': 1, 'minute': 0, 'vwap': 2., 'amount': 2., 'volume': 1.}])
        with patch.object(new, 'repair', task), patch.object(new, 'load_source_day', return_value=(raw, '/raw/15min.feather')):
            with self.assertRaisesRegex(ValueError, 'Unexplained'):
                new.run_task()
        self.procs['fit'].save_dump.assert_not_called()
        self.training.assert_not_called()
        self.assertIn('turn_fl', self.procs['fit'].load_dump().feature)
        self.assertEqual(self.report()['frames'][0]['inf_audit']['status'], 'failed')

    def test_new_task_rebuilds_healthy_dumps_over_available_history(self):
        spec = importlib.util.spec_from_file_location('repair_available_history', SCRIPT.with_name('repair_15mraw_inf.py'))
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        for frame, proc in self.procs.items():
            self.write(make_block(), frame)  # Prior sanitized dumps must not be skipped.
            proc.load_with_extension.side_effect = lambda p=proc, **kw: make_block().slice_date(p.load_start, p.load_end)
        self.procs['predict'].load_start = 20240120
        with patch.object(new, 'repair', task), patch.object(new.DB, 'dates', return_value=np.arange(20240102, 20240131)):
            new.run_task()
        for proc in self.procs.values():
            proc.load_with_extension.assert_called_once_with(reconstruct=True)
        self.assertEqual(self.procs['fit'].load_dump().date[0], 20240102)
        fit = self.report()['frames'][0]
        self.assertEqual(fit['source_date_coverage']['omitted_leading_dates'], [20240101])
        self.assertEqual(fit['rebuilt_date_coverage']['missing_dates'], [])
        self.assertTrue(Path(fit['backup']).exists())
        self.training.assert_called_once_with(new.REQUEST_ID)

    def test_prediction_source_failure_stops_before_either_frame_is_rebuilt(self):
        spec = importlib.util.spec_from_file_location('repair_missing_source', SCRIPT.with_name('repair_15mraw_inf.py'))
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        for frame in self.procs:
            self.write(make_block(), frame)
        with patch.object(new, 'repair', task), patch.object(new.DB, 'dates', return_value=np.arange(20240102, 20240131)):
            with self.assertRaisesRegex(ValueError, 'predict.*raw 15min dates missing'):
                new.run_task()
        for proc in self.procs.values():
            proc.load_with_extension.assert_not_called()
            proc.save_dump.assert_not_called()
        self.training.assert_not_called()
        report = self.report()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['frames'][1]['source_date_coverage']['missing_dates'], [20240101])


if __name__ == '__main__':
    main()
