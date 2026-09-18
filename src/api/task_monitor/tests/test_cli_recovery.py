from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.task_monitor import cli_recovery as recovery


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'state.json'
        self.state = {'desired': True, 'environment': {'DISPLAY': ':1', 'XDG_SESSION_ID': 'c2'},
                      'root': '/project', 'python': '/venv/python', 'process': {}, 'failures': 0}
        recovery.save(self.path, self.state)
        for name, value in [('sys.platform', 'linux')]:
            mock = patch('src.api.task_monitor.cli_recovery.' + name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_dispatch_requires_registration_and_does_not_duplicate(self):
        with patch.object(recovery, 'session_active', return_value=True), patch.object(recovery, 'dispatch') as dispatch:
            self.assertEqual(recovery.tick(self.path, now=100)['status'], 'recovering')
            self.assertEqual(recovery.tick(self.path, now=120)['status'], 'waiting-for-registration')
            dispatch.assert_called_once()
            recovery.tick(self.path, now=161)
            state = recovery.load(self.path)
            self.assertEqual(state['failures'], 1)
            self.assertGreater(state['next_attempt'], 161)

    def test_quit_reload_duplicate_and_pid_reuse(self):
        current = {'pid': 123, 'created': 3}
        with patch.dict('os.environ', {'DISPLAY': ':1'}), patch.object(recovery, 'process_identity', return_value=current):
            self.assertTrue(recovery.register(Path('/project'), self.path))
            recovery.stop(reload=True, path=self.path)
            self.assertTrue(recovery.load(self.path)['desired'])
            self.assertTrue(recovery.register(Path('/project'), self.path))
            recovery.stop(path=self.path)
            self.assertFalse(recovery.load(self.path)['desired'])
        process = MagicMock()
        process.create_time.return_value = 4
        with patch.object(recovery.psutil, 'Process', return_value=process):
            self.assertFalse(recovery.live(current))
        with patch.dict('os.environ', {'DISPLAY': ':1'}), patch.object(recovery, 'live', return_value=True), patch.object(recovery, 'process_identity', return_value={'pid': 999, 'created': 7}):
            self.assertFalse(recovery.register(Path('/project'), self.path))

    def test_login_absent_defers_without_consuming_retries(self):
        with patch.object(recovery, 'session_active', return_value=False), patch.object(recovery, 'dispatch') as dispatch:
            self.assertEqual(recovery.tick(self.path, now=100)['status'], 'waiting-for-desktop-login')
            dispatch.assert_not_called()
            self.assertEqual(recovery.load(self.path)['failures'], 0)

    def test_three_failures_pause_and_alert_retries(self):
        with patch.object(recovery, 'session_active', return_value=True), patch.object(recovery, 'dispatch', side_effect=RuntimeError('no user bus')):
            for now in (100, 300, 600):
                recovery.tick(self.path, now=now)
        self.assertTrue(recovery.load(self.path)['failure_paused'])
        sender = MagicMock(side_effect=[False, True])
        self.assertFalse(recovery.deliver(sender, self.path))
        self.assertTrue(recovery.deliver(sender, self.path))
        self.assertTrue(recovery.deliver(sender, self.path))
        self.assertEqual(sender.call_count, 2)

    def test_launch_uses_existing_window_and_cold_start(self):
        with patch.object(recovery.subprocess, 'run') as run:
            run.side_effect = [subprocess.CompletedProcess([], 0, json.dumps([{'window_id': 7}]), ''), MagicMock()]
            recovery.launch(self.path)
            self.assertIn('spawn', run.call_args.args[0])
            self.assertIn('7', run.call_args.args[0])
            run.side_effect = [subprocess.CompletedProcess([], 1, '', ''), MagicMock()]
            recovery.launch(self.path)
            self.assertIn('--always-new-process', run.call_args.args[0])

    def test_desktop_probe_checks_user_and_graphical_type(self):
        with patch.object(recovery.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'User=999999\nState=active\nType=x11\n')):
            self.assertFalse(recovery.session_active({'XDG_SESSION_ID': 'c2'}))
        self.assertFalse(recovery.session_active({}))

    def test_manual_additional_hub_is_allowed_but_automatic_duplicate_is_not(self):
        from src.api.calls.launcher import DirectCallHub
        with patch.object(recovery, 'register', return_value=False), patch.object(recovery, 'stop'), patch.object(DirectCallHub, 'run') as run:
            DirectCallHub.go()
            run.assert_called_once()
            run.reset_mock()
            DirectCallHub.go(_recovery=True)
            run.assert_not_called()

    def test_login_refreshes_environment_and_pause_prevents_dispatch(self):
        with patch.dict('os.environ', {'DISPLAY': ':8', 'XDG_SESSION_ID': 'c9'}, clear=True), patch.object(recovery, 'session_active', return_value=True), patch.object(recovery, 'dispatch') as dispatch:
            with patch.object(sys := recovery.sys, 'argv', ['cli', 'pause', '--state', str(self.path)]):
                recovery.main()
            with patch.object(sys, 'argv', ['cli', 'login', '--state', str(self.path)]):
                recovery.main()
            self.assertEqual(recovery.load(self.path)['environment']['DISPLAY'], ':8')
            dispatch.assert_not_called()
            with patch.object(sys, 'argv', ['cli', 'resume', '--state', str(self.path)]):
                recovery.main()
            dispatch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
