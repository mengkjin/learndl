"""Exercise migrations with real Git, subprocesses and a disposable host ledger."""
from __future__ import annotations

from contextlib import closing, nullcontext
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.automation import version_autofix as version
from src.automation.scheduling import channels, git_update, priority
from src.automation.scheduling.autofix_store import Store, initialize
from src.automation.scheduling import worker_bootstrap

__all__ = ['VersionAutofixTest']


class VersionAutofixTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve() / 'project'
        self.root.mkdir()
        self.runtime = self.root.parent / 'runtime'
        self.scheduling = self.runtime / 'scheduling'
        self.scheduling.mkdir(parents=True)
        self.git('init', '-b', 'master')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.write('.gitignore', '/models/\n/data/\n')
        self.release('0.7.1')
        self.commit()
        version.initialize(self.runtime)
        self.notify = Mock()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.DEVNULL)

    def write(self, name, body):
        file = self.root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(body)
        return file

    def release(self, number):
        self.write(version.VERSION_FILE, f'__version__ = {number!r}\n')

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '--allow-empty', '-m', 'test')

    def patch_file(self, release, name, body):
        return self.write(f'{version.PATCH_DIRECTORY}/{release}/{name}.py', body)

    def apply(self, **kwargs):
        with version.code_lease(self.scheduling) as fd:
            return version.apply_pending(self.root, self.runtime, pass_fds=(fd,), notify=self.notify, **kwargs)

    def control(self):
        control = self.root.parent / 'control'
        initialize(control, self.root)
        for group in channels.GROUPS:
            (control / f'channel-{group}.lock').touch()
        (control / 'reservation.lock').touch()
        store = Store(control)
        priority.initialize(store)
        channels.initialize(store)
        return store

    def git_worker(self, store):
        config = {'scheduler_version': 3, 'version_autofix': True}
        self.enterContext(patch.object(git_update, 'runtime_dir', return_value=self.scheduling))
        self.enterContext(patch.object(git_update, 'training_blocker', return_value=None))
        self.enterContext(patch('src.automation.scheduling.maintenance_gate.permit', side_effect=lambda **_: nullcontext()))
        self.enterContext(patch('src.automation.scheduling.autofix_store.state_dir', return_value=store.directory))
        self.enterContext(patch('src.automation.scheduling.config.installed_config', return_value=config))
        self.enterContext(patch.object(priority, 'enabled', return_value=True))

    def test_model_rename_and_data_move_across_versions(self):
        self.release('0.7.11')
        self.write('models/example.json', '{"old": 4}')
        self.write('data/old/item', 'payload')
        self.patch_file('0.7.2', '001_rename', """
import json

def apply(ctx):
    path = ctx.root / 'models/example.json'
    raw = path.read_text()
    config = json.loads(raw)
    if 'old' in config and 'new' not in config:
        (ctx.backup_dir / path.name).write_text(raw)
        config['new'] = config.pop('old')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(config))
        temp.replace(path)

def verify(ctx):
    return json.loads((ctx.root / 'models/example.json').read_text()) == {'new': 4}
""")
        self.patch_file('0.7.10', '001_move', """
def apply(ctx):
    assert ctx.source_version == '0.7.2'
    (ctx.root / 'data/old').rename(ctx.root / 'data/new')

def verify(ctx):
    return (ctx.root / 'data/new/item').read_text() == 'payload'
""")
        self.commit()
        self.assertIsNotNone(version.admission_reason(self.root, self.runtime))
        result = self.apply()
        self.assertEqual(result['patches'], ['0.7.2/001', '0.7.10/001'])
        self.assertEqual(result['version'], '0.7.11')
        self.assertEqual(self.apply()['patches'], [])
        self.assertIsNone(version.admission_reason(self.root, self.runtime))
        self.assertEqual(len(version.status(self.root, self.runtime)['records']), 2)
        self.assertEqual(len(list(version.ledger_path(self.runtime).parent.glob('attempts/*/backup/example.json'))), 1)

    def test_failed_verification_blocks_until_retry_and_keeps_attempts(self):
        self.release('0.7.2')
        source = """
def apply(ctx):
    path = ctx.root / 'data/written'
    path.parent.mkdir(exist_ok=True)
    path.write_text('done')

def verify(ctx):
    return %s
"""
        path = self.patch_file('0.7.2', '001_apply', source % 'False')
        self.commit()
        self.assertEqual(self.apply()['status'], 'blocked')
        self.assertIsNotNone(version.admission_reason(self.root, self.runtime))
        self.assertEqual(self.apply()['status'], 'blocked')
        self.assertEqual(self.notify.call_args_list[0], self.notify.call_args_list[1])
        path.write_text(source % "(ctx.root / 'data/written').read_text() == 'done'")
        self.commit()
        self.assertEqual(self.apply(retry='0.7.2/001')['status'], 'ready')
        records = version.status(self.root, self.runtime)['records']
        self.assertEqual([row['status'] for row in records], ['failed', 'success'])
        self.assertNotEqual(records[0]['digest'], records[1]['digest'])
        path.unlink()
        self.assertIn('Successful patch changed or disappeared', version.admission_reason(self.root, self.runtime) or '')

    def test_timeout_and_interrupted_worker_preserve_evidence(self):
        self.release('0.7.2')
        self.patch_file('0.7.2', '001_sleep', 'import time\ndef apply(ctx): time.sleep(60)\ndef verify(ctx): return True\n')
        self.write(f'{version.PATCH_DIRECTORY}/0.7.2/timeouts.json', '{"001_sleep.py": 1}')
        self.commit()
        self.assertIn('exceeded its 1s', self.apply()['reason'])
        with closing(version.connect(self.runtime)) as db, db:
            db.execute("UPDATE attempts SET status='running', error=NULL, finished=NULL")
        self.assertIn('manual retry', self.apply()['reason'])
        self.assertEqual(self.notify.call_args.args[0]['status'], 'interrupted')

    def test_version_local_support_is_verified_and_is_not_an_entrypoint(self):
        self.release('0.7.2')
        support = self.patch_file('0.7.2', 'support/helper', 'VALUE = "verified"\n')
        self.patch_file('0.7.2', '001_use_support', """
from src.automation.version_autofix import load_patch_support

def apply(ctx):
    helper = load_patch_support('0.7.2', 'helper', root=ctx.root)
    (ctx.root / 'proof.txt').write_text(helper.VALUE)

def verify(ctx):
    return (ctx.root / 'proof.txt').read_text() == 'verified'
""")
        self.commit()
        support.write_text('VALUE = "modified"\n')
        with self.assertRaisesRegex(version.MigrationError, 'Patch input differs from committed source'):
            self.apply()
        self.assertFalse((self.root / 'proof.txt').exists())
        support.write_text('VALUE = "verified"\n')
        self.assertEqual(self.apply()['patches'], ['0.7.2/001'])
        self.assertEqual((self.root / 'proof.txt').read_text(), 'verified')

    def test_committed_patch_hidden_by_working_tree_deletion_is_not_skipped(self):
        self.release('0.7.2')
        path = self.patch_file('0.7.2', '001_test', 'def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.commit()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.apply()
        self.assertIsNotNone(version.admission_reason(self.root, self.runtime))

    def test_uninitialized_baseline_downgrade_and_missing_ledger(self):
        self.release('0.7.2')
        self.commit()
        self.assertEqual(self.apply(baseline='0.7.2')['status'], 'initialized')
        self.release('0.7.1')
        self.assertIn('older than data version', version.admission_reason(self.root, self.runtime) or '')
        version.ledger_path(self.runtime).unlink()
        self.assertIn('check failed', version.admission_reason(self.root, self.runtime) or '')
        with self.assertRaises(version.sqlite3.Error):
            version.initialize(self.runtime, required=True)

    def test_manual_upgrade_runs_when_fetch_disabled_and_defers_on_code_lock(self):
        self.git_worker(self.control())
        self.release('0.7.2')
        self.commit()
        with version.code_lease(self.scheduling):
            self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'deferred')
        self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'ready')
        self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'disabled')

    def test_auto_update_defers_patch_to_checkout_verification(self):
        self.apply()
        store = self.control()
        self.git_worker(store)
        origin = self.root.parent / 'origin'
        self.git('clone', str(self.root), str(origin))
        self.git('remote', 'add', 'origin', str(origin))
        (origin / version.VERSION_FILE).write_text("__version__ = '0.7.2'\n")
        location = origin / version.PATCH_DIRECTORY / '0.7.2' / '001_patch.py'
        location.parent.mkdir(parents=True)
        location.write_text('def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.git('-C', str(origin), 'add', '.')
        self.git('-C', str(origin), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'upgrade')
        with patch.object(git_update, 'queue_update_email', return_value='notice'):
            result = git_update._channel_update(self.root, {'enabled': True})
        self.assertEqual(result['status'], 'updated')
        self.assertTrue(result['checkout_attempted'])
        self.assertEqual(priority.read(store)['phase'], 'checkout')
        self.assertEqual(version.status(self.root, self.runtime)['records'], [])
        self.assertEqual(git_update._channel_update(self.root, {'enabled': True})['status'], 'verified')
        self.assertIsNone(version.admission_reason(self.root, self.runtime))
        self.assertEqual(priority.read(store)['phase'], 'idle')

    def test_bootstrap_holds_new_business_before_import_after_manual_pull(self):
        control = self.control().directory
        (control / 'worker-settings.json').write_text(json.dumps({'root': str(self.root), 'runtime': str(self.scheduling)}))
        (self.scheduling / 'installed.json').write_text('{"scheduler_version": 3, "version_autofix": true}')
        self.release('0.7.2')
        self.commit()
        self.addCleanup(os.chdir, Path.cwd())
        with (patch.object(worker_bootstrap, '__file__', str(control / 'worker_bootstrap.py')),
              patch.object(sys, 'argv', ['bootstrap', 'lane', 'daily_update']),
              patch.object(sys, 'path', sys.path.copy()),
              patch.object(worker_bootstrap.runpy, 'run_module') as load):
            worker_bootstrap.main()
            load.assert_not_called()
            # The bootstrap releases the lock when it declines the start.
            self.apply()
            worker_bootstrap.main()
            load.assert_called_once()

    def test_verified_git_checkout_stays_blocked_on_failure_but_allows_mail(self):
        store = self.control()
        control = store.directory
        head = self.git('rev-parse', 'HEAD').decode().strip()
        priority.write(store, {'phase': 'checkout', 'last_completed': 0, 'target': head})
        (control / 'worker-settings.json').write_text(json.dumps({'runtime': str(self.scheduling)}))
        (self.scheduling / 'installed.json').write_text('{"scheduler_version": 3}')
        with (patch.object(priority, 'enabled', return_value=True),
              patch('src.automation.scheduling.autofix_store.state_dir', return_value=control),
              patch.object(git_update, 'repository_blocker', return_value=None)):
            result = git_update._admitted_update(self.root, {}, migrate=lambda: {'status': 'blocked', 'reason': 'test'})
            self.assertEqual(result['status'], 'blocked')
            self.assertTrue(worker_bootstrap.wait_for_code(control, 'lane'))
            self.assertFalse(worker_bootstrap.wait_for_code(control, 'email_outbox'))
            result = git_update._admitted_update(self.root, {}, migrate=lambda: {'status': 'ready'})
            self.assertEqual(result['status'], 'verified')
            self.assertFalse(worker_bootstrap.wait_for_code(control, 'lane'))

    def test_failure_mail_is_deduplicated_and_retried(self):
        row = version.AttemptRecord(id='attempt', patch='0.7.2/001', commit_hash='abc', host='test',
                                    attempt=1, status='failed', log_path='/tmp/patch.log',
                                    started=1, finished=2, error='verification failed', path='patch.py', digest='test')
        sender = Mock(side_effect=[False, True])
        with patch.object(git_update, 'runtime_dir', return_value=self.scheduling):
            git_update.queue_version_failure(row)
            git_update.queue_version_failure(row)
            self.assertFalse(git_update.deliver_update_emails(sender))
            self.assertTrue(git_update.deliver_update_emails(sender))
            self.assertTrue(git_update.deliver_update_emails(sender))
        self.assertEqual(sender.call_count, 2)
        self.assertIn('retry 0.7.2/001', sender.call_args.args[1])

    def test_retry_can_adopt_corrected_commit_after_timer_sees_head_change(self):
        store = self.control()
        self.git_worker(store)
        self.release('0.7.2')
        script = 'def apply(ctx): pass\ndef verify(ctx): return %s\n'
        file = self.patch_file('0.7.2', '001_apply', script % 'False')
        self.commit()
        original = self.git('rev-parse', 'HEAD').decode().strip()
        priority.write(store, {'phase': 'checkout', 'last_completed': 0, 'target': original})
        self.assertEqual(git_update._channel_update(self.root, {})['status'], 'blocked')
        self.assertEqual(priority.read(store)['version_verified_head'], original)

        file.write_text(script % 'True')
        self.commit()
        self.assertEqual(git_update._channel_update(self.root, {})['status'], 'blocked')
        self.assertFalse(priority.read(store)['version_verified'])
        with version.code_lease(self.scheduling) as fd:
            with self.assertRaisesRegex(version.MigrationError, 'first failed'):
                version.verify_manual_checkout(self.root, self.runtime, retry='0.7.2/002')
            self.assertEqual(priority.read(store)['target'], original)
            version.verify_manual_checkout(self.root, self.runtime, retry='0.7.2/001')
            version.apply_pending(self.root, self.runtime, pass_fds=(fd,), notify=self.notify, retry='0.7.2/001')
        self.assertEqual(git_update._channel_update(self.root, {})['status'], 'verified')
