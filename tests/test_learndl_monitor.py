"""Regression tests for monitor lifecycle guards."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.learndl_monitor import services

__all__ = []


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = patch.object(services, 'STATE', Path(self.temporary.name))
        self.state.start()
        self.addCleanup(self.state.stop)
        self.manager = services.Manager()

    def test_opening_console_does_not_start_services(self):
        with patch.object(self.manager, 'discover', return_value=[]), patch.object(self.manager, 'start') as start:
            self.assertTrue(all(self.manager.snapshot(key)['state'] == 'stopped' for key in services.SERVICES))
            start.assert_not_called()

    def test_duplicate_start_is_rejected(self):
        with patch.object(self.manager, 'discover', return_value=[MagicMock()]), \
                patch.object(services.threading, 'Thread', ImmediateThread), patch.object(self.manager, 'start') as start:
            self.manager.submit('tasks', 'start', None)
            start.assert_not_called()
            self.assertIn('already exists', self.manager.errors['tasks'])

    def test_reused_pid_cannot_be_stopped(self):
        process = MagicMock(pid=123)
        process.create_time.return_value = 200
        with patch.object(self.manager, 'discover', return_value=[process]), \
                patch.object(services.threading, 'Thread', ImmediateThread):
            self.manager.submit('tasks', 'stop', 123, 100)
            process.terminate.assert_not_called()
            self.assertIn('identity changed', self.manager.errors['tasks'])

    def test_parallel_operations_are_rejected(self):
        self.manager.locks['tasks'].acquire()
        with self.assertRaisesRegex(ValueError, 'already running'):
            self.manager.submit('tasks', 'start', None)
        self.manager.locks['tasks'].release()

    def test_invalid_service_and_action(self):
        for key, action in [('unknown', 'start'), ('tasks', 'scan'), ([], 'stop')]:
            with self.assertRaises(ValueError):
                self.manager.submit(key, action, None)

if __name__ == '__main__':
    unittest.main()
