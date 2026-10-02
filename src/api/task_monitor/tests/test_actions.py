"""Operator actions never signal an unconfirmed or replaced process."""
import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import psutil

from src.api.task_monitor.actions import KillTarget, kill_task, kill_unavailable_reason, prepare_kill
from src.api.task_monitor.core import TaskSnapshot


def snapshot(**changes):
    task = TaskSnapshot('job@100', 'job.py', 'python job.py', 100, 'running', 'script',
                        424242, 100, None, None, None, None)
    return replace(task, **changes)


class MonitorActionsTest(unittest.TestCase):
    def setUp(self):
        self.target = KillTarget('job@100', 424242, 99, 'job.py', 100)
        self.item = MagicMock(id='job@100', pid=424242, is_running=True)
        self.item.kill.return_value = True
        self.db = MagicMock()
        self.db.get_task.return_value = self.item
        self.proc = MagicMock()
        self.proc.status.return_value = 'sleeping'
        self.proc.create_time.return_value = 99
        self.db_patch = patch('src.api.task_monitor.actions.TaskDatabase', return_value=self.db)
        self.proc_patch = patch('src.api.task_monitor.actions.psutil.Process', return_value=self.proc)
        self.db_patch.start()
        self.proc_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(self.proc_patch.stop)

    def test_prepare_captures_identity_without_killing(self):
        self.assertEqual(prepare_kill(snapshot()), self.target)
        self.item.kill.assert_not_called()
        self.db.get_task.assert_not_called()

    def test_success_persists_operator_origin_and_binds_database(self):
        result = kill_task(self.target)
        self.assertEqual(result.status, 'success')
        self.item.kill.assert_called_once_with()
        self.item.set_task_db.assert_called_once_with(self.db)
        values = self.item.update.call_args.args[0]
        self.assertEqual(values['status'], 'killed')
        self.assertEqual(values['exit_error'], 'Killed by operator from Learndl Monitor')
        self.assertEqual(values['exit_code'], 1)
        self.assertIsInstance(values['end_time'], float)
        self.assertTrue(self.item.update.call_args.kwargs['sync'])

    def test_existing_killed_metadata_is_preserved(self):
        def kill():
            self.item.is_running = False
            return True
        self.item.kill.side_effect = kill
        self.assertEqual(kill_task(self.target).status, 'success')
        self.item.update.assert_not_called()

    def test_failure_never_marks_killed(self):
        self.item.kill.return_value = False
        self.assertEqual(kill_task(self.target).status, 'failed')
        self.item.update.assert_not_called()

    def test_changed_or_finished_record_is_rejected(self):
        for attr, value in [('id', 'other@100'), ('pid', 777777), ('is_running', False)]:
            with self.subTest(attr=attr):
                previous = getattr(self.item, attr)
                setattr(self.item, attr, value)
                self.assertEqual(kill_task(self.target).status, 'invalid')
                setattr(self.item, attr, previous)
        self.db.get_task.return_value = None
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.item.kill.assert_not_called()

    def test_reused_or_dead_pid_is_rejected(self):
        self.proc.create_time.return_value = 101
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.create_time.return_value = 99
        self.proc.status.return_value = 'zombie'
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.proc.status.side_effect = psutil.NoSuchProcess(424242)
        self.assertEqual(kill_task(self.target).status, 'invalid')
        self.item.kill.assert_not_called()

    def test_permission_denied_is_reported_without_status_update(self):
        self.proc.create_time.side_effect = psutil.AccessDenied(424242)
        self.assertEqual(kill_task(self.target).status, 'failed')
        self.item.kill.assert_not_called()
        self.item.update.assert_not_called()

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
        self.item.kill.assert_not_called()


class MonitorActionDatabaseTest(unittest.TestCase):
    def test_success_moves_a_real_record_from_running_to_finished(self):
        import tempfile
        from pathlib import Path
        from src.api.util.backend.task import TaskDatabase, TaskItem
        from src.api.task_monitor.core import TaskMonitorRepository
        from src.proj import PATH

        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / 'tasks.db'
            with patch.object(TaskDatabase, 'get_db_path', return_value=db_path):
                db = TaskDatabase()
                task = TaskItem(str(PATH.scpt / 'monitor_test.py'), status='running',
                                create_time=100, start_time=100, pid=424242).set_task_db(db)
                task.dump()
                repository = TaskMonitorRepository(db_path, Path(directory))
                self.assertEqual(repository.list_tasks(statuses={'running'}, window='history').total, 1)
                with patch('src.api.task_monitor.actions.psutil.Process') as process, \
                     patch.object(TaskItem, 'kill', return_value=True) as kill:
                    process.return_value.status.return_value = 'running'
                    process.return_value.create_time.return_value = 99
                    result = kill_task(KillTarget(task.id, 424242, 99, task.script, 100))
                self.assertEqual(result.status, 'success')
                kill.assert_called_once()
                self.assertEqual(repository.list_tasks(statuses={'running'}, window='history').total, 0)
                finished = repository.list_tasks(statuses={'killed'}, window='history')
                self.assertEqual(finished.total, 1)
                self.assertIn('Learndl Monitor', finished.tasks[0].exit_error)
