from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.api.task_monitor import diagnostic_sampler as sampler
from src.api.task_monitor import interactive
from src.api.util.backend.task import TaskDatabase


class InteractiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dbpatch = patch.object(TaskDatabase, 'get_db_path', return_value=self.root / 'tasks.db')
        dbpatch.start()
        self.addCleanup(dbpatch.stop)
        runpatch = patch.object(interactive, 'runtime_dir', return_value=self.root / 'runs')
        runpatch.start()
        self.addCleanup(runpatch.stop)
        self.processor = SimpleNamespace(key='minc', frame='fit', load_start=20100101, load_end=20260918)

    def test_success_records_files_before_compute_and_no_email(self):
        with patch.object(interactive, 'start_sampler', return_value={'mode': 'test'}):
            with interactive.reconstruction(self.processor) as task_id:
                task = TaskDatabase().get_task(task_id)
                self.assertEqual(task.status, 'running')
                self.assertTrue(task.exit_files)
                print('persisted before completion')
                log = next((self.root / 'runs').glob('*/output.log'))
                self.assertIn('persisted before completion', log.read_text())
        self.assertEqual(TaskDatabase().get_task(task_id).status, 'complete')
        sender = MagicMock()
        self.assertTrue(interactive.maintain(TaskDatabase(), sender))
        sender.assert_not_called()

    def test_exception_is_durable_retried_and_deduplicated(self):
        with patch.object(interactive, 'start_sampler', return_value={'mode': 'test'}):
            with self.assertRaisesRegex(ValueError, 'test failure'):
                with interactive.reconstruction(self.processor) as task_id:
                    raise ValueError('test failure')
        self.assertEqual(TaskDatabase().get_task(task_id).status, 'error')
        sender = MagicMock(side_effect=[False, True])
        self.assertFalse(interactive.maintain(TaskDatabase(), sender))
        self.assertTrue(list((self.root / 'runs').glob('*/pending.json')))
        self.assertTrue(interactive.maintain(TaskDatabase(), sender))
        self.assertTrue(interactive.maintain(TaskDatabase(), sender))
        self.assertEqual(sender.call_count, 2)
        self.assertIn('test failure', sender.call_args.args[1])

    def test_killed_run_evidence_and_retention(self):
        with patch.object(interactive, 'start_sampler', return_value={'mode': 'test'}):
            with interactive.reconstruction(self.processor) as task_id:
                print('last useful stage')
        db = TaskDatabase()
        db.update_task(task_id, status='killed', end_time=1.0)
        sender = MagicMock(return_value=False)
        with patch.object(interactive, 'evidence', return_value={'kernel': {'unavailable': 'permission denied'}}), patch.object(interactive, 'alive', return_value=False):
            self.assertFalse(interactive.maintain(db, sender, now=40 * 86400))
            self.assertTrue(list((self.root / 'runs').glob('*/output.log')))
            sender.return_value = True
            self.assertTrue(interactive.maintain(db, sender, now=40 * 86400))
            self.assertFalse(list((self.root / 'runs').glob('*/output.log')))

    def test_tee_keeps_logging_after_terminal_disappears_and_tail_is_bounded(self):
        original = MagicMock()
        original.write.side_effect = BrokenPipeError()
        output = io.StringIO()
        tee = interactive.Tee(original, output)
        tee.write('survived\n')
        self.assertEqual(output.getvalue(), 'survived\n')
        path = self.root / 'large.log'
        path.write_text('x' * 20000)
        self.assertEqual(len(interactive.tail(path)), 8192)

    def test_real_sampler_survives_target_sigkill(self):
        folder = self.root / 'sampler'
        folder.mkdir()
        target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        worker = None
        try:
            process = sampler.identity(target.pid)
            worker = subprocess.Popen([sys.executable, str(Path(sampler.__file__)), str(folder),
                                       str(target.pid), str(process['created'])])
            deadline = time.monotonic() + 10
            while not (folder / 'memory.jsonl').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue((folder / 'memory.jsonl').exists())
            target.kill()
            target.wait(timeout=5)
            worker.wait(timeout=15)
            self.assertEqual(worker.returncode, 0)
            self.assertTrue((folder / 'evidence.json').exists())
            lines = (folder / 'memory.jsonl').read_text().splitlines()
            self.assertIn('available', json.loads(lines[0]))
        finally:
            for process in (target, worker):
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)

    def test_killed_reconstruction_is_reconciled_with_flushed_output(self):
        folder = self.root / 'runs'
        code = f'''
import time
from pathlib import Path
from types import SimpleNamespace
from src.api.util.backend.task import TaskDatabase
from src.api.task_monitor import interactive
TaskDatabase.get_db_path = staticmethod(lambda: Path({str(self.root / 'tasks.db')!r}))
interactive.runtime_dir = lambda: Path({str(folder)!r})
interactive.start_sampler = lambda *args: {{'mode': 'test'}}
processor = SimpleNamespace(key='minc', frame='fit', load_start=20100101, load_end=20260918)
with interactive.reconstruction(processor):
    print('before kill', flush=True)
    time.sleep(30)
'''
        child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            log = None
            while time.monotonic() < deadline:
                log = next(folder.glob('*/output.log'), None)
                if log and 'before kill' in log.read_text():
                    break
                time.sleep(.05)
            self.assertIsNotNone(log)
            self.assertIn('before kill', log.read_text())
            child.kill()
            child.wait(timeout=5)
            db = TaskDatabase()
            changed = db.reconcile_stopped_tasks()
            self.assertEqual(len(changed), 1)
            self.assertEqual(next(iter(changed.values()))['status'], 'killed')
            sender = MagicMock(return_value=True)
            with patch.object(interactive, 'evidence', return_value={'unavailable': 'test'}):
                self.assertTrue(interactive.maintain(db, sender))
                self.assertTrue(interactive.maintain(db, sender))
            sender.assert_called_once()
            self.assertIn('before kill', sender.call_args.args[1])
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
