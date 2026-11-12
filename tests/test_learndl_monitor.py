"""Regression tests for monitor lifecycle guards and storage email defaults."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.api.learndl_monitor import services, storage

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

    def test_first_snapshot_no_email_but_rescan_sends_email(self):
        root = Path(self.temporary.name)
        report = MagicMock()
        report.return_value.run.return_value = {'report': root / 'snapshot.html', 'email_sent': True}
        snapshot = MagicMock()
        fake_modules = {
            'src.proj': SimpleNamespace(PATH=SimpleNamespace(production=root)),
            'src.api.calls.computer': SimpleNamespace(ProjectStorageReport=report),
            'src.proj.util.filesys.storage_browser': SimpleNamespace(StorageReportServer=snapshot),
        }

        class FakeHTTP:
            def __init__(self, address, handler):
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def serve_forever(self):
                handler = object.__new__(self.handler)
                handler.headers = {'Host': '127.0.0.1:8503'}
                handler.path = '/test-token/scan'
                handler.send_response = MagicMock()
                handler.end_headers = MagicMock()
                handler.do_POST()
                handler.send_response.assert_called_once_with(202)

        with patch.dict('sys.modules', fake_modules), patch.object(storage, 'STATE', root), \
                patch.object(storage, 'ThreadingHTTPServer', FakeHTTP), \
                patch.object(storage.secrets, 'token_urlsafe', return_value='test-token'), \
                patch.object(storage.threading, 'Thread', ImmediateThread):
            storage.main()
        self.assertEqual([call.kwargs['email'] for call in report.call_args_list], [False, True])


if __name__ == '__main__':
    unittest.main()
