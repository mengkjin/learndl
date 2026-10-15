"""Autofix sequencing, backups and fail-fast behavior without real database writes."""
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

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/.autofix/create_open_vwap_datablocks.py'
spec = importlib.util.spec_from_file_location('price_datablock_autofix', SCRIPT)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


class PriceDatablockAutofixTest(TestCase):
    def test_mature_endpoints(self):
        plan = task.make_plan(20240229)
        self.assertEqual(len(plan), 6)
        for job in plan:
            self.assertEqual(int(job['dates'][0]), 20050104)
            self.assertLessEqual(task.CALENDAR.td(int(job['dates'][-1]), job['days'] + 1).as_int(), 20240229)
            self.assertTrue(job['key'].endswith(('_open', '_vwap')))

    def test_backup_copies_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'old.mmap').mkdir()
            (root / 'old.mmap/values').write_text('original')
            (root / 'raw.feather').write_text('raw')
            (root / 'backup').mkdir()
            records = task.backup_existing([root / 'old.mmap', root / 'raw.feather', root / 'absent'], root / 'backup')
            self.assertEqual((Path(records[0]['backup']) / 'values').read_text(), 'original')
            self.assertEqual(Path(records[1]['backup']).read_text(), 'raw')
            self.assertIsNone(records[2]['backup'])

    def run_fixture(self, *, failure=None, dry_run=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            events = []
            jobs = [{'price': price, 'days': days, 'key': f'ret{days}_{price}', 'dates': np.array([20240102])}
                    for price in ('open', 'vwap') for days in (5, 10, 20)]
            stack.enter_context(patch.object(task, 'PATH', SimpleNamespace(main=root)))
            stack.enter_context(patch.object(task, 'make_plan', return_value=jobs))
            stack.enter_context(patch.object(task.DB, 'DBPath', side_effect=lambda src, key: SimpleNamespace(parent=root / 'raw' / key)))
            stack.enter_context(patch.object(task.DataBlock, 'path_preprocess', side_effect=lambda source, frame: root / 'dumps' / f'{source}_{frame}'))
            stack.enter_context(patch.object(task.DB, 'dates', return_value=[20240102]))
            stack.enter_context(patch.object(task, 'calc_price_labels', side_effect=lambda date, days, price: None if failure == 'missing' else pd.DataFrame({'secid': [1], f'rtn_{days}_{price}': [0.1]})))
            def save(df, src, key, date, **kwargs):
                events.append(key)
                return failure != 'save'
            stack.enter_context(patch.object(task.DB, 'save', side_effect=save))
            def processor(source, frame, **kwargs):
                price = source.removeprefix('y_')
                features = [f'{kind}_{days}_{price}' for kind in ('rtn', 'std') for days in (5, 10, 20)]
                block = DataBlock(torch.ones(1, 1, 1, 6), np.array([1]), np.array([20240102]), np.array(features))
                proc = Mock()
                proc.load_with_extension.return_value = block
                proc.load_dump.return_value = block
                proc.save_dump.side_effect = lambda block: events.append(f'{source}/{frame}')
                return proc
            factory = stack.enter_context(patch.object(task.PrePros, 'get_processor', side_effect=processor))
            if failure:
                with self.assertRaises((ValueError, RuntimeError)):
                    task.run_task()
                factory.assert_not_called()
            else:
                task.run_task(dry_run=dry_run)
            reports = list(root.rglob('report.json'))
            if dry_run:
                self.assertEqual(reports, [])
                self.assertEqual(events, [])
                factory.assert_not_called()
            else:
                report = json.loads(reports[0].read_text())
                self.assertEqual(report['status'], 'failed' if failure else 'success')
                if not failure:
                    self.assertEqual(events, [job['key'] for job in jobs] + ['y_open/fit', 'y_vwap/fit', 'y_open/predict', 'y_vwap/predict'])
                    self.assertEqual(report['completed'], events)

    def test_raw_labels_precede_all_four_blocks(self):
        self.run_fixture()

    def test_missing_labels_and_failed_saves_stop_before_preprocessing(self):
        for failure in ('missing', 'save'):
            with self.subTest(failure=failure):
                self.run_fixture(failure=failure)

    def test_dry_run_does_not_write(self):
        self.run_fixture(dry_run=True)


if __name__ == '__main__':
    main()
