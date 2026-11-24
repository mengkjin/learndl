"""Regressions for monitor launch targeting and native service entrypoints."""
from __future__ import annotations

__all__ = [
    'MonitorLaunchTest',
]
import unittest
from unittest.mock import MagicMock, patch

from src.api.calls.app import LaunchTaskMonitor
from src.proj.util.shell.linux.wezterm.open import WezTermOpener


class MonitorLaunchTest(unittest.TestCase):
    def test_monitor_requests_tab_without_new_workspace(self):
        with patch('src.proj.util.shell.Shell.open') as launch:
            LaunchTaskMonitor().run()
        self.assertEqual(launch.call_args.kwargs['new_on'], 'tab')
        self.assertNotIn('as_from_workspace', launch.call_args.kwargs)

    def test_current_window_reused_when_socket_heuristic_misses(self):
        module = 'src.proj.util.shell.linux.wezterm.open.'
        opener = WezTermOpener.__new__(WezTermOpener)
        opener._available = True
        result = MagicMock(stdout='[{"pane_id": 13, "window_id": 7}]')
        with patch(module + 'discover_wezterm_gui_socket', return_value=None), \
             patch(module + 'subprocess.run', return_value=result), \
             patch(module + 'activate_wezterm'), patch(module + 'bring_wezterm_to_foreground_soon'), \
             patch(module + 'process.popen_detached') as spawn, \
             patch.dict('os.environ', {'WEZTERM_PANE': '13'}):
            opener.run('echo monitor', new_on='tab')
        argv = spawn.call_args.args[0]
        self.assertEqual(argv[:5], ['wezterm', 'cli', 'spawn', '--window-id', '7'])
        self.assertNotIn('--new-window', argv)

    def test_monitor_command_starts_native_service(self):
        with patch('src.proj.util.shell.Shell.open') as launch:
            LaunchTaskMonitor().run()
        self.assertIn('python -m src.api.task_monitor.launch --open', launch.call_args.args[0])
