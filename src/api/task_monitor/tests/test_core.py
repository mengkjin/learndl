from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.api.task_monitor.core import TaskMonitorRepository, is_background_source, read_live_log


class TaskMonitorRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / 'tasks.db'
        connection = sqlite3.connect(self.db_path)
        with connection:
            connection.execute('''CREATE TABLE task_records (
                task_id TEXT PRIMARY KEY, script TEXT, cmd TEXT, create_time REAL, status TEXT,
                source TEXT, pid INTEGER, start_time REAL, end_time REAL, exit_code INTEGER,
                exit_message TEXT, exit_error TEXT
            )''')
            connection.execute('CREATE TABLE task_exit_files (id INTEGER PRIMARY KEY, task_id TEXT, file_path TEXT)')
            connection.executemany('INSERT INTO task_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
                ('active-old', 'a.py', 'python a.py', 1, 'running', 'bash', None, 1, None, None, None, None),
                ('recent-error', 'b.py', 'python b.py', 2, 'error', 'script', None, 2, 99_000, 1, None, 'boom'),
                ('old-complete', 'c.py', 'python c.py', 3, 'complete', 'script', None, 3, 1, 0, None, None),
            ])
        connection.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_lists_active_and_recent_finished_tasks_only(self) -> None:
        tasks = TaskMonitorRepository(self.db_path, self.root).list_tasks(now=100_000, lookback_hours=1)
        self.assertEqual([task.task_id for task in tasks], ['active-old', 'recent-error'])
        self.assertTrue(tasks[0].is_background)

    def test_crash_tail_is_plain_text_and_bounded(self) -> None:
        output = self.root / 'running.md'
        output.write_text(
            '# Header\n- <u>one &amp; two</u>  \n\n- 09:19:01.193: ^\n'
            '- 09:19:01.194: ^\n- <span>last</span>\n', encoding='utf-8',
        )
        self.assertEqual(
            read_live_log(output, max_lines=4),
            '# Header\n- one & two  \n… 3 blank/control-output lines suppressed …\n- last',
        )

    def test_background_source_detection(self) -> None:
        self.assertTrue(is_background_source('bash'))
        self.assertTrue(is_background_source('SystemD'))
        self.assertFalse(is_background_source('script'))
