"""Operator actions never signal an unconfirmed or replaced process."""
from __future__ import annotations

__all__ = [
    'snapshot',
    'MonitorActionsTest',
    'MonitorActionDatabaseTest',
]
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from src.api.task_monitor.actions import KillTarget, kill_task, kill_unavailable_reason, prepare_kill
from src.api.task_monitor.core import TaskSnapshot
from src.automation.scheduling import runtime


def snapshot(**changes):
    task = TaskSnapshot('job@100', 'job.py', 'python job.py', 100, 'running', 'script',
                        424242, 100, None, None, None, None)
    return replace(task, **changes)


class MonitorActionsTest(unittest.TestCase):
    def setUp(self):
        self.target = KillTarget('job@100', 424242, 99, 'job.py', 100)
        self.item = MagicMock(id='job@100', pid=424242, is_running=True, exit_files=[])
        self.item.get_crash_protector.return_value = []
        self.db = MagicMock()
        self.db.get_task.return_value = self.item
        self.proc = MagicMock()
        self.proc.status.return_value = 'sleeping'
        self.proc.create_time.return_value = 99
        self.proc.wait.return_value = -15
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.directory = Path(directory)
        self.enterContext(patch.object(runtime, 'runtime_dir', return_value=self.directory))
        self.db_patch = patch('src.api.task_monitor.actions.TaskDatabase', return_value=self.db)
        self.proc_patch = patch('src.api.task_monitor.actions.psutil.Process', return_value=self.proc)
        self.db_patch.start()
        self.proc_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(self.proc_patch.stop)

    def test_prepare_captures_identity_without_killing(self):
        self.assertEqual(prepare_kill(snapshot()), self.target)
        self.proc.terminate.assert_not_called()
        self.db.get_task.assert_not_called()

    def test_managed_training_uses_scheduler_group_stop(self):
        identity={'pid':424242,'process_created':99,'cgroup':'/test'}
        task=snapshot(task_id='managed:scheduler:one',managed_kind='scheduler',process_created=99)
        with patch('src.automation.scheduling.operator_stop.prepare',return_value=identity) as prepare, \
             patch('src.automation.scheduling.operator_stop.stop',return_value='Stop requested') as stop:
            self.assertIsNone(kill_unavailable_reason(task))
            target=prepare_kill(task)
            self.assertEqual(target.managed_identity,identity)
            self.assertEqual(kill_task(target).status,'success')
            prepare.assert_called_once_with(task.task_id,424242,99)
            stop.assert_called_once_with(task.task_id,identity)
        self.db.get_task.assert_not_called()
        self.proc.terminate.assert_not_called()

    def test_success_persists_operator_origin_and_queues_mail(self):
        result = kill_task(self.target)
        self.assertEqual(result.status, 'success')
        self.proc.terminate.assert_called_once_with()
        self.proc.wait.assert_called_once_with(timeout=2)
        self.db.update_task.assert_called_once()
        values = self.db.update_task.call_args.kwargs
        self.assertEqual(values['status'], 'killed')
        self.assertEqual(values['exit_error'], 'Killed by operator from Learndl Monitor')
        self.assertEqual(values['exit_code'], -15)
        self.assertIsInstance(values['end_time'], float)
        self.assertTrue(values['backend_updated'])
        sender = MagicMock(return_value=True)
        self.assertTrue(runtime.deliver_timeout_events(sender))
        sender.assert_called_once()
        self.assertIn('No Crash Protector output', sender.call_args.args[1])

    def test_exit_log_survives_removal_and_is_retried_once(self):
        log = self.directory / 'crash.md'
        log.write_text('before stop\n')
        self.item.get_crash_protector.return_value = [log]
        self.item.exit_files = ['previous.log']

        def wait(**kwargs):
            self.assertIn(self.target.task_id, runtime.timeout_task_ids())
            sender = MagicMock()
            self.assertTrue(runtime.deliver_timeout_events(sender))
            sender.assert_not_called()
            with log.open('a') as output:
                output.write('during shutdown\n')
            log.unlink()
            return -15

        self.proc.wait.side_effect = wait
        self.assertEqual(kill_task(self.target).status, 'success')
        attachment, previous = self.db.update_task.call_args.kwargs['exit_files']
        self.assertEqual(previous, 'previous.log')
        self.assertEqual(Path(attachment).read_text(), 'before stop\nduring shutdown\n')
        sender = MagicMock(side_effect=[False, True])
        self.assertFalse(runtime.deliver_timeout_events(sender))
        self.assertTrue(runtime.deliver_timeout_events(sender))
        self.assertTrue(runtime.deliver_timeout_events(sender))
        self.assertEqual(sender.call_count, 2)
        self.assertEqual(sender.call_args_list[0], sender.call_args_list[1])
        self.assertEqual(sender.call_args.kwargs['attachments'], [Path(attachment)])

    def test_failure_never_marks_killed(self):
        self.proc.wait.side_effect = psutil.TimeoutExpired(2)
        self.assertEqual(kill_task(self.target).status, 'failed')
        self.proc.kill.assert_called_once_with()
        self.db.update_task.assert_not_called()
        self.assertNotIn(self.target.task_id, runtime.timeout_task_ids())
        sender = MagicMock()
        self.assertTrue(runtime.deliver_timeout_events(sender))
        sender.assert_not_called()

    def test_watchdog_recovers_email_if_monitor_exits_during_stop(self):
        store = runtime.RunStore()
        attachment = self.directory / 'retained.md'
        attachment.write_text('last line\n')
        run = dict(id=f'{runtime.OPERATOR_TASK_PREFIX}{self.target.task_id}', task=self.target.script,
                   pid=self.target.pid, process_created=self.target.process_created, stop_requested_at=100,
                   operator_crash_logs={self.target.task_id: [str(attachment)]})
        store.event(run, runtime.OPERATOR_STOP_EVENT, 'Operator stop', awaiting_exit=True)
        self.proc.status.return_value = psutil.STATUS_ZOMBIE
        sender = MagicMock(return_value=True)
        with patch.object(runtime.time, 'time', return_value=120):
            self.assertTrue(runtime.deliver_timeout_events(sender, store))
        self.assertEqual(sender.call_args.kwargs['attachments'], [attachment])

    def test_changed_or_finished_record_is_rejected(self):
        for attr, value in [('id', 'other@100'), ('pid', 777777), ('is_running', False)]:
            with self.subTest(attr=attr):
                previous = getattr(self.item, attr)
                setattr(self.item, attr, value)
                self.assertEqual(kill_task(self.target).status, 'invalid')
                setattr(self.item, attr, previous)
        self.db.get_task.return_value = None
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.terminate.assert_not_called()

    def test_reused_or_dead_pid_is_rejected(self):
        self.proc.create_time.return_value = 101
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.create_time.return_value = 99
        self.proc.status.return_value = 'zombie'
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.status.side_effect = psutil.NoSuchProcess(424242)
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.terminate.assert_not_called()

    def test_permission_denied_is_reported_without_status_update(self):
        self.proc.create_time.side_effect = psutil.AccessDenied(424242)
        self.assertEqual(kill_task(self.target).status, 'failed')
        self.proc.terminate.assert_not_called()
        self.db.update_task.assert_not_called()

    def test_disabled_targets(self):
        for changes in ({'pid': None}, {'pid': os.getpid()}, {'pid': 0}, {'status': 'complete'}):
            with self.subTest(changes=changes):
                task = snapshot(**changes)
                self.assertIsNotNone(kill_unavailable_reason(task))
                with self.assertRaises(ValueError):
                    prepare_kill(task)
        self.proc.status.side_effect = psutil.NoSuchProcess(424242)
        self.assertIsNotNone(kill_unavailable_reason(snapshot()))
        self.assertEqual(kill_task(replace(self.target, pid=os.getpid())).status, 'invalid')
        self.proc.terminate.assert_not_called()


class MonitorActionDatabaseTest(unittest.TestCase):
    def test_success_moves_a_real_record_from_running_to_finished(self):
        import tempfile
        from pathlib import Path
        from src.api.util.backend.task import TaskDatabase, TaskItem
        from src.api.task_monitor.core import TaskMonitorRepository
        from src.proj import PATH

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / 'tasks.db'
            with patch.object(TaskDatabase, 'get_db_path', return_value=db_path), \
                 patch.object(runtime, 'runtime_dir', return_value=Path(directory)):
                db = TaskDatabase()
                task = TaskItem(str(PATH.scpt / 'monitor_test.py'), status='running',
                                create_time=100, start_time=100, pid=424242).set_task_db(db)
                task.dump()
                repository = TaskMonitorRepository(db_path, Path(directory))
                self.assertEqual(repository.list_tasks(statuses={'running'}, window='history').total, 1)
                with patch('src.api.task_monitor.actions.psutil.Process') as process:
                    process.return_value.status.return_value = 'running'
                    process.return_value.create_time.return_value = 99
                    process.return_value.wait.return_value = -15
                    result = kill_task(KillTarget(task.id, 424242, 99, task.script, 100))
                self.assertEqual(result.status, 'success')
                process.return_value.terminate.assert_called_once_with()
                self.assertEqual(repository.list_tasks(statuses={'running'}, window='history').total, 0)
                finished = repository.list_tasks(statuses={'killed'}, window='history')
                self.assertEqual(finished.total, 1)
                exit_error = finished.tasks[0].exit_error
                if exit_error is None:
                    raise AssertionError('killed task has no exit error')
                self.assertIn('Learndl Monitor', exit_error)
