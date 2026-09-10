from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.api.task_monitor.watchdog import JobResult, WatchdogJob, run_watchdog
from src.api.util.backend.task import TaskDatabase, TaskItem
from src.proj import PATH


class _FakeTaskDatabase:
    def __init__(self, changed: dict[str, dict[str, Any]], tasks: dict[str, TaskItem]) -> None:
        self.changed = changed
        self.tasks = tasks
        self.due_crash_logs: list[tuple[str, Path]] = []

    def reconcile_stopped_tasks(self) -> dict[str, dict[str, Any]]:
        changed, self.changed = self.changed, {}
        return changed

    def get_task(self, task_id: str) -> TaskItem | None:
        return self.tasks.get(task_id)

    def get_killed_tasks_since(self, since: float) -> list[TaskItem]:
        return [task for task in self.tasks.values() if task.status == 'killed' and task.end_time and task.end_time >= since]

    def recovered_crash_logs_due(self) -> list[tuple[str, Path]]:
        return self.due_crash_logs

    def prune_recovered_crash_logs(self, *, cached_paths: set[Path]) -> list[Path]:
        return list(cached_paths)


class TaskWatchdogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp_dir.name) / 'state.json'
        self.task = TaskItem(
            '/project/scripts/train.py', cmd='python train.py', create_time=1,
            status='killed', pid=123, start_time=100.0,
            exit_error='process disappeared', task_id='train.py@1',
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, database: _FakeTaskDatabase, sender: Any, *, now: float, **kwargs: Any) -> bool:
        return run_watchdog(
            task_db=database, units=[], state_path=self.state_path, email_sender=sender,
            now=now, **kwargs,
        )

    def test_newly_killed_task_sends_once_and_retries_delivery(self) -> None:
        db = _FakeTaskDatabase({'train.py@1': {'status': 'killed'}}, {'train.py@1': self.task})
        attempts = 0

        def send(*args: Any, **kwargs: Any) -> bool:
            nonlocal attempts
            attempts += 1
            return attempts > 1

        self.assertFalse(self._run(db, send, now=1_000))
        self.assertEqual(json.loads(self.state_path.read_text())['alerts']['pending_tasks'], ['train.py@1'])
        self.assertTrue(self._run(db, send, now=1_060))
        self.assertEqual(attempts, 2)
        self.assertEqual(json.loads(self.state_path.read_text())['alerts']['pending_tasks'], [])

    def test_unit_alert_resets_only_after_recovery(self) -> None:
        db = _FakeTaskDatabase({}, {})
        emails = 0
        active = False

        def check_unit(unit: str) -> tuple[bool, str]:
            return active, 'active' if active else 'failed'

        def send(*args: Any, **kwargs: Any) -> bool:
            nonlocal emails
            emails += 1
            return True

        for now in (1_000, 1_060):
            self.assertTrue(run_watchdog(
                task_db=db, units=['learndl-app.service'], state_path=self.state_path,
                email_sender=send, now=now, unit_checker=check_unit,
            ))
        self.assertEqual(emails, 1)
        active = True
        self.assertTrue(run_watchdog(
            task_db=db, units=['learndl-app.service'], state_path=self.state_path,
            email_sender=send, now=1_120, unit_checker=check_unit,
        ))
        active = False
        self.assertTrue(run_watchdog(
            task_db=db, units=['learndl-app.service'], state_path=self.state_path,
            email_sender=send, now=1_180, unit_checker=check_unit,
        ))
        self.assertEqual(emails, 2)

    def test_cache_job_is_due_every_fifteen_minutes_and_prewarms_before_pruning(self) -> None:
        db = _FakeTaskDatabase({}, {})
        db.due_crash_logs = [('task', Path('/tmp/crash.md'))]
        cache = MagicMock()
        cache.ensure_both.return_value = True
        cache.cleanup.return_value = {'remaining_bytes': 10}
        send = MagicMock(return_value=True)
        self.assertTrue(self._run(db, send, now=1_000, cache=cache))
        self.assertTrue(self._run(db, send, now=1_060, cache=cache))
        self.assertTrue(self._run(db, send, now=1_900, cache=cache))
        self.assertEqual(cache.ensure_both.call_count, 2)
        state = json.loads(self.state_path.read_text())
        self.assertEqual(state['jobs']['task_monitor_cache']['stats']['prewarmed_crash_logs'], 1)

    def test_job_failure_is_recorded_without_blocking_other_jobs(self) -> None:
        db = _FakeTaskDatabase({}, {})
        runs: list[str] = []

        def broken(_: object) -> JobResult:
            raise RuntimeError('broken')

        def healthy(_: object) -> JobResult:
            runs.append('healthy')
            return JobResult(stats={'ok': True})

        jobs = (WatchdogJob('broken', 60, broken), WatchdogJob('healthy', 60, healthy))
        self.assertFalse(self._run(db, lambda *args, **kwargs: True, now=1_000, jobs=jobs))
        self.assertEqual(runs, ['healthy'])
        state = json.loads(self.state_path.read_text())
        self.assertIn('RuntimeError: broken', state['jobs']['broken']['last_error'])
        self.assertEqual(state['jobs']['healthy']['stats'], {'ok': True})

    def test_watchdog_state_migrates_legacy_alerts(self) -> None:
        self.state_path.write_text(json.dumps({'pending_tasks': ['train.py@1'], 'unit_alerted': {}}))
        db = _FakeTaskDatabase({}, {'train.py@1': self.task})
        self.assertTrue(self._run(db, lambda *args, **kwargs: True, now=1_000))
        state = json.loads(self.state_path.read_text())
        self.assertEqual(state['version'], 2)
        self.assertIn('jobs', state)
        self.assertEqual(state['alerts']['pending_tasks'], [])

    def test_dead_active_record_is_killed_without_crash_file(self) -> None:
        database_path = Path(self.temp_dir.name) / 'tasks.db'
        with patch.object(TaskDatabase, 'get_db_path', return_value=database_path):
            database = TaskDatabase()
            task = TaskItem(str(PATH.scpt / 'train.py'), cmd='python train.py', create_time=2, status='running', pid=999_999, start_time=2)
            database.new_task(task)
            with patch('src.api.util.backend.task.process.check_status', return_value='complete'), patch.object(TaskItem, 'get_crash_protector', return_value=[]):
                changed = database.reconcile_stopped_tasks()
        self.assertEqual(changed[task.id]['status'], 'killed')


if __name__ == '__main__':
    unittest.main()
