"""Real local Git repositories; no remote service, training or email required."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil

from src.api.task_monitor.scheduling import git_update as updater
from src.api.task_monitor.watchdog import JobResult, WatchdogJob, run_watchdog
from src.proj import PATH
from src.proj.util.script import FitLock


class GitUpdateTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source'
        self.checkout = self.root / 'server'
        self.source.mkdir()
        self.git(self.source, 'init', '-b', 'master')
        self.git(self.source, 'config', 'user.email', 'test@example.invalid')
        self.git(self.source, 'config', 'user.name', 'Test')
        self.commit('code.py', 'VALUE = 1\n')
        self.git(self.root, 'clone', str(self.source), str(self.checkout))
        self.git(self.checkout, 'config', 'user.email', 'test@example.invalid')
        self.git(self.checkout, 'config', 'user.name', 'Test')
        self.before = self.git(self.checkout, 'rev-parse', 'HEAD')
        self.commit('code.py', 'VALUE = 2\n')
        self.after = self.git(self.source, 'rev-parse', 'HEAD')
        self.busy = self.enterContext(patch.object(updater, 'training_blocker', return_value=None))
        self.enterContext(patch.object(updater, 'runtime_dir', return_value=self.root / 'runtime'))
        self.enterContext(patch('src.proj.MACHINE', SimpleNamespace(platform_server=True, name='test-server')))

    def git(self, directory, *args):
        return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(directory), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()

    def commit(self, filename, content):
        path = self.source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git(self.source, 'add', filename)
        self.git(self.source, 'commit', '-m', 'test update')

    def update(self):
        return updater.check_and_update({'max_seconds': 45}, root=self.checkout)

    def test_clean_master_fast_forwards_then_is_up_to_date(self):
        result = self.update()
        self.assertEqual(result['status'], 'updated')
        self.assertEqual((result['before'], result['after']), (self.before, self.after))
        self.assertEqual((self.checkout / 'code.py').read_text(), 'VALUE = 2\n')
        self.assertEqual(self.update()['status'], 'up_to_date')

    def test_success_mail_is_durable_retried_and_not_repeated(self):
        result = self.update()
        self.assertIn('notification_id', result)
        sender = Mock(side_effect=[False, OSError('SMTP unavailable'), True])
        self.assertFalse(updater.deliver_update_emails(sender))
        self.assertFalse(updater.deliver_update_emails(sender))
        self.assertTrue(updater.deliver_update_emails(sender))
        self.assertIn(self.before, sender.call_args.args[1])
        self.assertIn(self.after, sender.call_args.args[1])
        self.assertIn('code.py', sender.call_args.args[1])
        self.assertIn('test-server', sender.call_args.args[0])
        self.assertEqual(self.update()['status'], 'up_to_date')
        updater.queue_update_email(result, self.checkout)
        self.assertTrue(updater.deliver_update_emails(sender))
        self.assertEqual(sender.call_count, 3)

    def test_deferred_or_failed_pull_does_not_queue_success_mail(self):
        self.busy.return_value = 'training active'
        self.assertEqual(self.update()['status'], 'deferred')
        self.busy.return_value = None
        self.git(self.checkout, 'remote', 'set-url', 'origin', str(self.root / 'missing'))
        self.assertEqual(self.update()['status'], 'error')
        sender = Mock()
        self.assertTrue(updater.deliver_update_emails(sender))
        sender.assert_not_called()

    def test_dirty_staged_untracked_and_in_progress_are_preserved(self):
        path = self.checkout / 'code.py'
        path.write_text('LOCAL = 1\n')
        self.assertEqual(self.update()['status'], 'deferred')
        self.git(self.checkout, 'add', 'code.py')
        self.assertEqual(self.update()['status'], 'deferred')
        self.assertEqual(path.read_text(), 'LOCAL = 1\n')
        self.git(self.checkout, 'reset', '--hard', self.before)  # Test fixture only.
        extra = self.checkout / 'notes.txt'
        extra.write_text('keep me')
        self.assertEqual(self.update()['status'], 'deferred')
        extra.unlink()
        marker = self.checkout / '.git' / 'MERGE_HEAD'
        marker.write_text(self.after)
        self.assertEqual(self.update()['status'], 'deferred')
        self.assertTrue(marker.exists())
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)

    def test_local_commit_branch_and_detached_head_never_rewritten(self):
        self.git(self.checkout, 'checkout', '-b', 'experiment')
        self.assertEqual(self.update()['status'], 'deferred')
        self.git(self.checkout, 'checkout', 'master')
        (self.checkout / 'local.py').write_text('LOCAL = True\n')
        self.git(self.checkout, 'add', 'local.py')
        self.git(self.checkout, 'commit', '-m', 'local work')
        local = self.git(self.checkout, 'rev-parse', 'HEAD')
        self.assertIn('divergent', self.update()['reason'])
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), local)
        self.git(self.checkout, 'checkout', '--detach', local)
        self.assertEqual(self.update()['status'], 'deferred')

    def test_active_training_defers_and_finishing_allows_update(self):
        self.busy.return_value = 'training/evaluation pipeline is active'
        self.assertEqual(self.update()['status'], 'deferred')
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)
        self.busy.return_value = None
        self.assertEqual(self.update()['status'], 'updated')

    def test_training_start_or_local_edit_during_check_is_rechecked(self):
        self.busy.side_effect = [None, 'training started']
        self.assertEqual(self.update()['reason'], 'training started')
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)
        def edit():
            (self.checkout / 'code.py').write_text('do not overwrite\n')
            return None
        self.busy.side_effect = edit
        self.assertEqual(self.update()['status'], 'deferred')
        self.assertEqual((self.checkout / 'code.py').read_text(), 'do not overwrite\n')

    def test_ignored_local_file_is_not_overwritten_by_new_tracked_file(self):
        self.commit('.gitignore', 'secret.txt\n')
        self.git(self.checkout, 'fetch', 'origin')
        self.git(self.checkout, 'merge', '--ff-only', 'origin/master')
        (self.checkout / 'secret.txt').write_text('local secret')
        (self.source / 'secret.txt').write_text('new default')
        self.git(self.source, 'add', '-f', 'secret.txt')
        self.git(self.source, 'commit', '-m', 'track ignored file')
        self.assertEqual(self.update()['status'], 'error')
        self.assertEqual((self.checkout / 'secret.txt').read_text(), 'local secret')

    def test_hooks_do_not_run_and_environment_changes_are_reported(self):
        marker = self.root / 'hook-ran'
        hook = self.checkout / '.git/hooks/post-merge'
        hook.write_text(f'#!/bin/sh\ntouch "{marker}"\n')
        hook.chmod(0o700)
        self.commit('uv.lock', '# lock change\n')
        self.commit('runs/scheduling/maintenance.yaml', '# config change\n')
        result = self.update()
        self.assertEqual(result['status'], 'updated')
        self.assertEqual(len(result['follow_up']), 2)
        self.assertFalse(marker.exists())

    def test_shared_pipeline_guard_prevents_update_without_serializing_training(self):
        from src.proj.util.script.code_update_lock import training_code_guard
        with patch.object(PATH, 'runtime', self.root), patch.object(updater, 'runtime_dir', return_value=self.root / 'scheduling'):
            with training_code_guard(), training_code_guard():
                self.assertEqual(self.update()['status'], 'deferred')
                self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)
            self.assertEqual(self.update()['status'], 'updated')

    def test_rewritten_remote_master_does_not_rewind_local_code(self):
        self.assertEqual(self.update()['status'], 'updated')
        self.git(self.source, 'reset', '--hard', self.before)  # Simulate upstream rewind in the fixture.
        result = self.update()
        self.assertIn(result['status'], {'error', 'deferred'})
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.after)

    def test_network_timeout_is_bounded_and_preserves_head(self):
        original = updater._git
        def timed_out(root, args, deadline, **kwargs):
            if args[0] == 'fetch':
                raise updater.GitUpdateError('Git command timed out')
            return original(root, args, deadline, **kwargs)
        with patch.object(updater, '_git', side_effect=timed_out):
            result = self.update()
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['before'], self.before)
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)

    def test_unavailable_remote_does_not_change_checkout(self):
        self.git(self.checkout, 'remote', 'set-url', 'origin', str(self.root / 'missing'))
        self.assertEqual(self.update()['status'], 'error')
        self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), self.before)


class TrainingActivityTest(unittest.TestCase):
    def test_full_pipeline_and_old_fit_lock_are_checked(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(PATH, 'lc_machine', Path(temp)), \
             patch.object(FitLock, 'is_held', return_value=False) as held, patch.object(updater, 'RunStore') as store:
            store.return_value.active.return_value = []
            history = Path(temp) / 'training_history' / 'old-run' / 'run.json'
            history.parent.mkdir(parents=True)
            data = dict(status='running', pid=os.getpid(), process_created=psutil.Process().create_time())
            history.write_text(json.dumps(data))
            self.assertIn('pipeline', updater.training_blocker())
            data['status'] = 'success'
            history.write_text(json.dumps(data))
            self.assertIsNone(updater.training_blocker())
            held.return_value = True
            self.assertIn('fit lock', updater.training_blocker())
            held.return_value = False
            history.write_text('invalid json')
            self.assertIn('cannot be verified', updater.training_blocker())


class WatchdogUpdateOrderTest(unittest.TestCase):
    def test_update_is_after_mail_and_failure_respects_five_minutes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            order = []
            jobs = [WatchdogJob('git_auto_update', 300, lambda _: (order.append('git') or JobResult(stats={'status': 'error', 'reason': 'offline'}))),
                    WatchdogJob('maintenance', 60, lambda _: (order.append('maintenance') or JobResult()))]
            with patch('src.api.task_monitor.scheduling.config.installed_config', return_value=None), \
                 patch.object(updater, 'runtime_dir', return_value=root), \
                 patch('src.api.task_monitor.watchdog._deliver_alerts', side_effect=lambda **_: (order.append('alerts') or True)), \
                 patch('src.api.task_monitor.scheduling.runtime.deliver_timeout_events', side_effect=lambda _: (order.append('mail') or True)):
                args = dict(task_db=Mock(), cache=Mock(), units=[], state_path=root / 'state.json', email_sender=Mock(), jobs=jobs)
                self.assertFalse(run_watchdog(**args, now=1000))
                self.assertEqual(order, ['maintenance', 'alerts', 'mail', 'git'])
                order.clear()
                self.assertTrue(run_watchdog(**args, now=1060))
                self.assertNotIn('git', order)
                self.assertFalse(run_watchdog(**args, now=1300))

    def test_update_mail_attempted_after_pull_and_retried_on_non_update_tick(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = dict(status='updated', before='a' * 40, after='b' * 40, changed_files=['code.py'])
            def update(_):
                result['notification_id'] = updater.queue_update_email(result, root)
                return JobResult(stats=result)
            job = WatchdogJob('git_auto_update', 300, update)
            with patch.object(updater, 'runtime_dir', return_value=root), \
                 patch('src.api.task_monitor.scheduling.config.installed_config', return_value=None), \
                 patch('src.api.task_monitor.watchdog._deliver_alerts', return_value=True), \
                 patch('src.api.task_monitor.scheduling.runtime.deliver_timeout_events', return_value=True):
                send = Mock(side_effect=[False, True])
                args = dict(task_db=Mock(), cache=Mock(), units=[], state_path=root / 'state.json', email_sender=send, jobs=[job])
                self.assertFalse(run_watchdog(**args, now=1000))
                self.assertEqual(send.call_count, 1)  # Same tick as the successful pull.
                self.assertTrue(run_watchdog(**args, now=1060))
                self.assertTrue(run_watchdog(**args, now=1120))
                self.assertEqual(send.call_count, 2)  # Retry before the next Git check.


if __name__ == '__main__':
    unittest.main()
