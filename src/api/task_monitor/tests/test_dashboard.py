"""Streamlit interaction tests with fake processes and bounded temporary logs."""
from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.api.task_monitor.actions import KillResult, KillTarget
from src.api.task_monitor.core import TaskPage, WatchdogHealth
from src.api.task_monitor.resources import ResourceSnapshot
from src.api.task_monitor.tests.test_actions import snapshot


class MonitorDashboardTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.log = root / 'output.log'
        self.log.write_text('visible live output\n')
        self.old = snapshot(task_id='old@100', script='old.py', exit_files=(self.log,))
        self.new = snapshot(task_id='new@200', script='new.py', create_time=200, start_time=200,
                            exit_files=(self.log,))
        self.tasks = [self.new, self.old]
        repository = stack.enter_context(patch('src.api.task_monitor.launch.TaskMonitorRepository')).return_value
        self.repository = repository
        repository.list_tasks.side_effect = lambda **kw: TaskPage(tuple(self.tasks), len(self.tasks), None, True)
        repository.get_task.side_effect = lambda task_id: next((t for t in self.tasks if t.task_id == task_id), None)
        repository.watchdog_health.return_value = WatchdogHealth(100, False, {})
        repository.status_counts.return_value = {'running': 0, 'complete': 1, 'error': 0, 'killed': 0}
        stack.enter_context(patch('src.api.task_monitor.launch._resources',
                                 return_value=ResourceSnapshot(100, None, None, None, (), ())))
        stack.enter_context(patch('src.api.task_monitor.launch.kill_unavailable_reason', return_value=None))
        stack.enter_context(patch('src.api.task_monitor.launch.prepare_kill', side_effect=lambda t:
                                 KillTarget(t.task_id, t.pid, 99, t.script, t.effective_start)))
        self.kill = stack.enter_context(patch('src.api.task_monitor.launch.kill_task',
                                             return_value=KillResult('success', 'Killed target')))
        self.refresh = stack.enter_context(patch('streamlit_autorefresh.st_autorefresh'))

    def app(self, entry='_render_running'):
        app = AppTest.from_string(
            'import streamlit as st\n'
            'st.set_page_config(layout="wide")\n'
            f'from src.api.task_monitor.launch import {entry}\n{entry}()\n',
            default_timeout=10,
        ).run()
        self.assertFalse(app.exception)
        return app

    def test_default_dashboard_shows_latest_log_without_a_click(self):
        app = self.app('_render_dashboard')
        self.assertIn('new.py', [item.value for item in app.subheader])
        self.assertIn('visible live output', app.code[0].value)
        self.assertEqual(self.repository.list_tasks.call_args.kwargs['statuses'], {'running'})
        self.assertEqual(self.repository.list_tasks.call_args.kwargs['window'], 'history')
        self.refresh.assert_called_with(interval=3000, key='running-refresh')

    def test_pin_follow_and_finished_selection(self):
        app = self.app()
        app.button(key='view-old@100').click().run()
        self.tasks.insert(0, replace(self.new, task_id='newer@300', script='newer.py', start_time=300))
        app.run()
        self.assertIn('old.py', [item.value for item in app.subheader])
        app.button(key='running-follow').click().run()
        self.assertIn('newer.py', [item.value for item in app.subheader])
        app.button(key='view-old@100').click().run()
        self.tasks.remove(self.old)
        app.run()
        self.assertIn('newer.py', [item.value for item in app.subheader])
        self.assertTrue(any('has finished' in item.value for item in app.info))

    def test_empty_missing_and_appending_output(self):
        self.tasks.clear()
        app = self.app()
        self.assertTrue(any('No running tasks' in item.value for item in app.info))
        self.tasks.append(self.old)
        app.run()
        self.log.write_text('visible live output\nappended line\n')
        app.run()
        self.assertIn('appended line', app.code[0].value)
        self.log.unlink()
        app.run()
        self.assertTrue(any('No live output' in item.value for item in app.info))
        self.tasks[0] = replace(self.old, exit_files=())
        app.run()
        self.assertTrue(any('No output file' in item.value for item in app.info))

    def test_cancel_does_not_kill_and_restores_refresh(self):
        app = self.app()
        self.refresh.reset_mock()
        app.button(key='kill-old@100').click().run()
        self.assertFalse(app.exception)
        self.refresh.assert_not_called()
        self.assertTrue(next(b for b in app.button if b.label == 'Confirm Kill').disabled)
        app.checkbox[0].check().run()
        self.kill.assert_not_called()
        next(b for b in app.button if b.label == 'Cancel').click().run()
        self.kill.assert_not_called()
        self.refresh.assert_called_with(interval=3000, key='running-refresh')
        self.assertNotIn('monitor-kill-target', app.session_state)

    def test_confirm_keeps_target_and_executes_once(self):
        app = self.app()
        app.button(key='kill-old@100').click().run()
        self.assertFalse(app.checkbox[0].value)
        # A new task during confirmation cannot change the frozen target.
        self.tasks.insert(0, replace(self.new, task_id='newer@300', script='newer.py'))
        app.run()
        app.checkbox[0].check().run()
        next(b for b in app.button if b.label == 'Confirm Kill').click().run()
        self.assertFalse(app.exception)
        self.kill.assert_called_once()
        self.assertEqual(self.kill.call_args.args[0].task_id, self.old.task_id)
        self.assertNotIn('monitor-kill-target', app.session_state)
        # The consumed token prevents replay even before the next full browser rerun.
        from src.api.task_monitor.launch import _queue_kill
        with patch('src.api.task_monitor.launch.st.session_state', {}):
            _queue_kill('old-checkbox')
        self.kill.assert_called_once()

    def test_finished_has_only_terminal_filters_and_independent_state(self):
        self.tasks[:] = [replace(self.old, status='complete', end_time=150)]
        app = self.app('_render_finished')
        self.assertEqual(app.multiselect[0].options, ['complete', 'error', 'killed'])
        self.assertEqual(self.repository.list_tasks.call_args.kwargs['statuses'], {'complete', 'error', 'killed'})
        self.assertEqual(self.repository.list_tasks.call_args.kwargs['window'], 'history')
        self.refresh.assert_called_with(interval=30000, key='finished-refresh')
        app.session_state['running-pinned-task'] = 'retained'
        app.multiselect[0].set_value(['error']).run()
        self.assertEqual(app.session_state['running-pinned-task'], 'retained')
