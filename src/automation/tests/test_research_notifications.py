"""Research start mail and script-owned completion, without real mail or workers."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.automation.scheduling import runtime, recovery


class ResearchNotificationsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = runtime.RunStore(Path(temp.name) / 'runs.sqlite')
        self.enterContext(patch('src.api.util.backend.task.TaskDatabase'))
        self.enterContext(patch('src.automation.scheduling.idle_worklist.settings', return_value={}))

    def run_record(self, kind):
        return dict(id=kind, task=kind, task_kind=kind, owner='idle_worklist',
                    phase='running', started_at=1, timeout_seconds=None,
                    config={'recipient': 'research@example.invalid'},
                    revisions={'id': 'experiment', 'schema_version': 3, 'cleanup_legacy': True})

    def test_start_retries_same_message_and_delivers_once_for_each_kind(self):
        for kind in ('rl_experiment', 'factor_sequence'):
            with self.subTest(kind=kind):
                run = self.run_record(kind)
                self.store.put(run)
                self.store.event(run, 'started', 'Research process started.')
                sender = Mock(side_effect=[False, True])
                self.assertFalse(runtime.deliver_timeout_events(sender, self.store))
                first = sender.call_args
                self.assertTrue(runtime.deliver_timeout_events(sender, self.store))
                self.assertEqual(first, sender.call_args)
                self.assertIn('started', first.args[0])
                self.assertIn('schema_version: 3', first.args[1])
                if kind == 'rl_experiment':
                    self.assertEqual(first.kwargs['recipient'], 'research@example.invalid')
                self.store.event(run, 'started', 'Duplicate observation')
                runtime.deliver_timeout_events(sender, self.store)
                self.assertEqual(sender.call_count, 2)

    def test_handoff_suppresses_completion_even_without_smtp_receipt(self):
        for kind in ('rl_experiment', 'factor_sequence'):
            run = self.run_record(kind)
            self.store.put(run)
            recovery.record(self.store, run['id'], script_notification_handoff=True)
            for phase, code in [('complete', 0), ('error', 1), ('killed', -9)]:
                run.update(phase=phase, returncode=code)
                self.assertFalse(runtime.idle_exit_needs_email(self.store, run))

    def test_startup_failure_still_notifies_and_no_email_is_respected(self):
        run = self.run_record('rl_experiment')
        run.update(phase='error', returncode=1)
        self.store.put(run)
        self.assertTrue(runtime.idle_exit_needs_email(self.store, run))
        runtime.finish_idle_event(self.store, run, 'Script import failed')
        sender = Mock(return_value=True)
        runtime.deliver_timeout_events(sender, self.store)
        sender.assert_called_once()
        self.assertIn('Script import failed', sender.call_args.args[1])
        run = self.run_record('factor_sequence') | {'no_email': True}
        self.store.put(run)
        self.store.event(run, 'started', 'Started')
        sender.reset_mock()
        runtime.deliver_timeout_events(sender, self.store)
        sender.assert_not_called()
