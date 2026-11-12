"""The CLI opens the detached console in place and survives launch failures."""
from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

from src.api.calls.app import LaunchLearndlMonitor
from src.api.calls.launcher import DirectCallHub
from src.proj import PATH

__all__ = []


class MonitorCLITest(unittest.TestCase):
    def test_hub_opens_console_without_spawning_a_pane(self):
        with patch.object(LaunchLearndlMonitor, 'run') as run, \
                patch.object(LaunchLearndlMonitor, 'spawn_in_pane') as pane:
            DirectCallHub()._dispatch_top_level('Learndl Monitor')
        run.assert_called_once_with()
        pane.assert_not_called()

    def test_launcher_uses_current_interpreter_and_project_root(self):
        with patch('src.api.calls.app.subprocess.run') as execute:
            LaunchLearndlMonitor().run()
        execute.assert_called_once_with(
            [sys.executable, '-m', 'src.api.learndl_monitor'], cwd=PATH.main, check=True,
        )

    def test_launch_failure_returns_to_hub(self):
        for error in (OSError('unavailable'), subprocess.CalledProcessError(1, ['python'])):
            with self.subTest(error=error), \
                    patch('src.api.calls.app.subprocess.run', side_effect=error), \
                    patch('src.proj.Logger.error') as log:
                DirectCallHub()._dispatch_top_level('Learndl Monitor')
                log.assert_called_once()


if __name__ == '__main__':
    unittest.main()
