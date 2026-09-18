from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.task_monitor.core import TaskMonitorRepository, is_background_source, read_live_log
from src.api.util.backend.task import TaskDatabase, TaskItem


def _create_database(path: Path, rows: list[tuple]) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.execute('''CREATE TABLE task_records (
            task_id TEXT PRIMARY KEY, script TEXT, cmd TEXT, create_time REAL, status TEXT,
            source TEXT, pid INTEGER, start_time REAL, end_time REAL, exit_code INTEGER,
            exit_message TEXT, exit_error TEXT
        )''')
        connection.execute('CREATE TABLE task_exit_files (id INTEGER PRIMARY KEY, task_id TEXT, file_path TEXT)')
        connection.executemany('INSERT INTO task_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
    connection.close()


class TaskMonitorRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / 'tasks.db'
        _create_database(self.db_path, [
            ('starting', 'a.py', 'python a.py', 99_900, 'starting', 'bash', None, None, None, None, None, None),
            ('running', 'b.py', 'python b.py', 99_800, 'running', 'app', 10, 99_800, None, None, None, None),
            ('recent-error', 'c.py', 'python c.py', 99_000, 'error', 'script', None, 99_000, 99_500, 1, None, 'boom'),
            ('old-complete', 'd.py', 'python d.py', 1, 'complete', 'script', None, 1, 2, 0, None, None),
        ])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_filters_status_and_time_and_maps_starting_to_running(self) -> None:
        repository = TaskMonitorRepository(self.db_path, self.root)
        page = repository.list_tasks(statuses={'running'}, window='last_day', now=100_000)
        self.assertEqual([task.task_id for task in page.tasks], ['starting', 'running'])
        errors = repository.list_tasks(statuses={'error'}, window='last_day', now=100_000)
        self.assertEqual([task.task_id for task in errors.tasks], ['recent-error'])
        history = repository.list_tasks(statuses={'complete'}, window='history', now=100_000)
        self.assertEqual([task.task_id for task in history.tasks], ['old-complete'])
        empty = repository.list_tasks(statuses=set(), window='history', now=100_000)
        self.assertEqual(empty.tasks, ())
        self.assertEqual(empty.total, 0)

    def test_detail_loads_files_but_list_does_not(self) -> None:
        connection = sqlite3.connect(self.db_path)
        with connection:
            connection.execute('INSERT INTO task_exit_files VALUES (?, ?, ?)', (1, 'recent-error', '/tmp/report.html'))
        connection.close()
        repository = TaskMonitorRepository(self.db_path, self.root)
        page = repository.list_tasks(statuses={'error'}, window='history')
        self.assertEqual(page.tasks[0].exit_files, ())
        detail = repository.get_task('recent-error')
        assert detail is not None
        self.assertEqual(detail.exit_files, (Path('/tmp/report.html'),))

    def test_keyset_pagination_on_ten_thousand_records(self) -> None:
        large_path = self.root / 'large.db'
        rows = [
            (f'task-{index:05d}', 'a.py', 'python a.py', float(index), 'complete', 'script', None,
             float(index), float(index + 1), 0, None, None)
            for index in range(10_000)
        ]
        _create_database(large_path, rows)
        repository = TaskMonitorRepository(large_path, self.root)
        first = repository.list_tasks(window='history', limit=50)
        second = repository.list_tasks(window='history', limit=50, cursor=first.next_cursor)
        self.assertEqual(first.total, 10_000)
        self.assertEqual(len(first.tasks), 50)
        self.assertEqual(len(second.tasks), 50)
        self.assertTrue(set(task.task_id for task in first.tasks).isdisjoint(task.task_id for task in second.tasks))

    def test_live_tail_is_plain_and_compacts_control_noise(self) -> None:
        output = self.root / 'running.md'
        output.write_text(
            '# Header\n- <u>one &amp; two</u>  \n\n- 09:19:01.193: ^\n'
            '- 09:19:01.194: ^\n- <span style="color: red">last</span>\n', encoding='utf-8',
        )
        result = read_live_log(output, max_lines=4)
        self.assertNotIn('<span', result)
        self.assertIn('3 blank/control-output lines suppressed', result)
        self.assertIn('last', result)

    def test_watchdog_health(self) -> None:
        heartbeat = self.root / 'task_watchdog' / 'state.json'
        heartbeat.parent.mkdir()
        heartbeat.write_text(json.dumps({
            'jobs': {'task_lifecycle': {'last_success_at': 99_900}},
        }), encoding='utf-8')
        with patch('src.api.task_monitor.core.time.time', return_value=100_000):
            health = TaskMonitorRepository(self.db_path, self.root).watchdog_health('task_lifecycle', 200)
        self.assertFalse(health.stale)

    def test_background_source_detection(self) -> None:
        self.assertTrue(is_background_source('bash'))
        self.assertTrue(is_background_source('SystemD'))
        self.assertFalse(is_background_source('script'))

    def test_pid_reuse_is_treated_as_stopped(self) -> None:
        task = TaskItem('a.py', status='running', pid=123, start_time=100)
        with patch('src.api.util.backend.task.process.check_status', return_value='running'):
            with patch('src.api.util.backend.task.psutil.Process') as process_class:
                process_class.return_value.create_time.return_value = 1000
                self.assertTrue(TaskDatabase._recorded_process_stopped(task))

    def test_existing_cli_process_is_not_pid_reuse(self) -> None:
        task = TaskItem('a.py', status='running', pid=123, start_time=1000)
        with patch('src.api.util.backend.task.process.check_status', return_value='running'), patch('src.api.util.backend.task.psutil.Process') as process_class:
            process_class.return_value.create_time.return_value = 100
            self.assertFalse(TaskDatabase._recorded_process_stopped(task))

    def test_unavailable_process_probe_is_not_death(self) -> None:
        import psutil
        task = TaskItem('a.py', status='running', pid=123, start_time=1000)
        with patch('src.api.util.backend.task.process.check_status', side_effect=psutil.AccessDenied(123)):
            self.assertFalse(TaskDatabase._recorded_process_stopped(task))
        with patch('src.api.util.backend.task.process.check_status', return_value='running'), patch('src.api.util.backend.task.psutil.Process', side_effect=psutil.AccessDenied(123)):
            self.assertFalse(TaskDatabase._recorded_process_stopped(task))
