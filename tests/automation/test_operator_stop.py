"""Tests for operator stop requests and managed process termination."""
from __future__ import annotations

__all__ = [
    'OperatorStopTests',
]
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from src.automation.scheduling import runtime
from src.automation.scheduling.runtime import RunStore, timeout_message
from src.automation.scheduling import operator_stop


class OperatorStopTests(unittest.TestCase):
    def test_stop_persists_before_signal_and_survives_worker_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp) / "runs.sqlite")
            run = dict(
                id="one",
                phase="running",
                owner="idle_worklist",
                pid=123,
                process_created=4,
                runner_pid=122,
                uid=501,
                boot_id="boot",
                cgroup="/group",
            )
            store.put(run)
            store.put(dict(id="retry", phase="queued", parent_run_id="one"))
            with (
                patch.object(operator_stop, "store_for", return_value=store),
                patch.object(operator_stop, "members", return_value=[Mock()]),
                patch.object(operator_stop, "signal_members") as signal,
            ):
                identity = operator_stop.prepare("managed:scheduler:one", 123, 4)

                def check(*args):
                    self.assertTrue(store.get("one")["suppress_auto_recovery"])
                    self.assertEqual(store.get("retry")["phase"], "cancelled")

                signal.side_effect = check
                operator_stop.stop("managed:scheduler:one", identity)
                signal.assert_called_once()
                store.put(run)  # Late runner write must not remove the stop intent.
                self.assertTrue(store.get("one")["suppress_auto_recovery"])
                self.assertIn("Monitor", timeout_message(store.get("one")))
                run["pid"] = 987
                store.put(run)
                with self.assertRaises(ValueError):
                    operator_stop.stop("managed:scheduler:one", identity)
                self.assertEqual(signal.call_count, 1)

    def test_no_creation_of_missing_ledger(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(operator_stop, "runtime_dir", return_value=Path(tmp)),
        ):
            with self.assertRaises(ValueError):
                operator_stop.store_for("scheduler")
            self.assertFalse((Path(tmp) / "runs.sqlite").exists())

    def test_email_waits_for_group_exit_and_attaches_final_crash_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            store = RunStore(directory / 'runs.sqlite')
            store.put(dict(id='one', task='training', owner='idle_worklist', phase='running',
                           pid=123, process_created=4, started_at=5, timeout_seconds=None))
            store.link('train@5', 'one')
            log = directory / 'train.md'
            log.write_text('training\n')
            task = Mock(id='train@5', pid=123, exit_files=[])
            task.get_crash_protector.side_effect = lambda: [log] if log.exists() else []
            database = Mock()
            database.get_task.return_value = task

            def signal(*args):
                with log.open('a') as output:
                    output.write('shutdown\n')
                log.unlink()

            with (
                patch.object(operator_stop, 'store_for', return_value=store),
                patch.object(operator_stop, 'members', return_value=[Mock()]),
                patch.object(operator_stop, 'signal_members', side_effect=signal),
                patch('src.api.util.backend.task.TaskDatabase', return_value=database),
                patch.object(runtime, 'members', side_effect=[[Mock()], [], []]),
                patch.object(runtime, 'signal_members'),
            ):
                identity = operator_stop.prepare('managed:scheduler:one', 123, 4)
                operator_stop.stop('managed:scheduler:one', identity)
                runtime.inspect_timeouts(store)
                self.assertEqual(store.get('one')['phase'], 'terminating')
                with store.connect() as connection:
                    self.assertEqual(connection.execute('SELECT id FROM events').fetchall(), [('one:started',)])
                    connection.execute('UPDATE events SET sent=1 WHERE id=?', ('one:started',))
                runtime.inspect_timeouts(store)
                runtime.inspect_timeouts(store)
                self.assertEqual(store.get('one')['phase'], 'killed')
                sender = Mock(return_value=True)
                self.assertTrue(runtime.deliver_timeout_events(sender, store))
                self.assertTrue(runtime.deliver_timeout_events(sender, store))
                sender.assert_called_once()
                self.assertIn('Stopped by operator', sender.call_args.args[0])
                attachment, = sender.call_args.kwargs['attachments']
                self.assertEqual(attachment.read_text(), 'training\nshutdown\n')
                self.assertEqual(database.update_task.call_args.kwargs['exit_files'], [str(attachment)])
