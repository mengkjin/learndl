"""Exercise migrations with real Git, subprocesses and a disposable host ledger."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from functools import partial
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from src.automation import version_patch as version
from src.automation.scheduling import autofix, calendar_lanes, channels, git_update, priority, runner
from src.automation.scheduling.autofix_store import Store, initialize
from src.automation.scheduling import worker_bootstrap
from src.automation.scheduling import idle_worklist as idle
from src.automation.scheduling.runtime import RunStore

__all__ = ['VersionPatchTest']


class VersionPatchTest(unittest.TestCase):
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
        self.notify = Mock()
        version.initialize(self.runtime)
        self.apply(baseline='0.7.1')

    def use_fresh_runtime(self):
        self.runtime = self.root.parent / 'fresh-host' / 'runtime'
        self.scheduling = self.runtime / 'scheduling'
        self.scheduling.mkdir(parents=True)

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
        config = {'scheduler_version': 3, 'version_patch': True}
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
from src.automation.version_patch import load_patch_support

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
        self.use_fresh_runtime()
        version.initialize(self.runtime)
        self.release('0.7.2')
        self.commit()
        self.assertEqual(self.apply(baseline='0.7.2')['status'], 'initialized')
        self.release('0.7.1')
        self.assertIn('older than data version', version.admission_reason(self.root, self.runtime) or '')
        version.ledger_path(self.runtime).unlink()
        self.assertIn('check failed', version.admission_reason(self.root, self.runtime) or '')
        with self.assertRaises(version.sqlite3.Error):
            version.initialize(self.runtime, required=True)

    def test_fresh_server_automatically_skips_historical_patches(self):
        self.use_fresh_runtime()
        self.release('0.7.2')
        self.write('results/README.md', 'Checkout template')
        (self.root / 'results/.DS_Store').touch()
        (self.root / 'data/Interim/empty').mkdir(parents=True)
        (self.root / 'models/nn').mkdir(parents=True)
        outbox = self.runtime.parent / 'email_outbox'
        outbox.mkdir()
        (outbox / '.lock').touch()
        self.patch_file('0.7.2', '001_historical', 'raise AssertionError("historical patch ran")\n')
        self.commit()

        preview = version.status(self.root, self.runtime, required=False)
        self.assertEqual(preview['pending'], [])
        self.assertIsNone(preview['baseline'])
        self.assertFalse(version.ledger_path(self.runtime).exists())
        version.initialize(self.runtime)
        self.assertIn('awaiting automatic initialization', version.admission_reason(self.root, self.runtime) or '')
        self.git_worker(self.control())
        worker = git_update._channel_update(self.root, {'enabled': False})
        self.assertEqual(worker, {'status': 'ready', 'version': '0.7.2', 'patches': []})
        state = version.status(self.root, self.runtime)
        self.assertEqual(state['baseline'], '0.7.2')
        self.assertEqual(state['records'], [])
        self.assertIsNone(version.admission_reason(self.root, self.runtime))

        self.release('0.7.3')
        self.patch_file('0.7.3', '001_next', 'def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.commit()
        self.assertEqual(self.apply()['patches'], ['0.7.3/001'])

    def test_import_without_ledger_requires_explicit_data_version(self):
        self.use_fresh_runtime()
        self.release('0.7.2')
        data = self.write('models/nn/imported/metadata.json', '{}')
        self.patch_file('0.7.2', '001_upgrade', 'def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.commit()
        with self.assertRaisesRegex(version.MigrationError, 'init --baseline'):
            version.status(self.root, self.runtime, required=False)
        self.assertFalse(version.ledger_path(self.runtime).exists())

        version.initialize(self.runtime)
        self.assertIn('init --baseline', version.admission_reason(self.root, self.runtime) or '')
        with self.assertRaisesRegex(version.MigrationError, 'init --baseline'):
            self.apply()
        with closing(version.connect(self.runtime, readonly=True)) as db:
            self.assertEqual(version.read_state(db), (None, None))
        self.assertTrue(data.exists())
        self.assertEqual(self.apply(baseline='0.7.1')['status'], 'initialized')
        self.assertEqual(self.apply()['patches'], ['0.7.2/001'])

    def test_imported_host_artifacts_also_block_baseline_inference(self):
        self.use_fresh_runtime()
        for relative in ('optuna/imported.sqlite', 'email_outbox/.archive/imported.eml'):
            path = self.runtime.parent / relative
            path.parent.mkdir(parents=True)
            path.touch()
            with self.assertRaisesRegex(version.MigrationError, 'init --baseline'):
                version.status(self.root, self.runtime, required=False)
            path.unlink()

    def test_rename_reuses_legacy_history_and_preserves_successful_patch_identity(self):
        self.release('0.7.2')
        path = self.patch_file('0.7.2', '001_test', 'def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.commit()
        self.assertEqual(self.apply()['patches'], ['0.7.2/001'])
        legacy = self.runtime / 'version_autofix'
        recorded_path = 'scripts/.version_autofix/0.7.2/001_test.py'
        with closing(version.connect(self.runtime)) as db, db:
            db.execute('UPDATE attempts SET path=?', (recorded_path,))
        version.ledger_path(self.runtime).parent.rename(legacy)
        self.write('models/imported/metadata.json', '{}')

        self.assertTrue(version.activation({'version_autofix': True}))
        version.initialize(self.runtime, required=True)
        self.assertIsNone(version.admission_reason(self.root, self.runtime))
        self.assertEqual(self.apply()['patches'], [])
        self.assertEqual(version.status(self.root, self.runtime)['records'][0]['path'], recorded_path)
        self.assertFalse((self.runtime / 'version_patch').exists())
        path.write_text('def verify(ctx): return True\n')
        self.assertIn('Successful patch changed', version.admission_reason(self.root, self.runtime) or '')

        (self.runtime / 'version_patch').mkdir()
        with self.assertRaisesRegex(version.MigrationError, 'Both version patch state directories'):
            version.initialize(self.runtime, required=True)

    def test_shipped_072_scripts_match_successful_legacy_ledger(self):
        self.release('0.7.2')
        scripts = version.discover(version.ROOT, '0.7.2')
        digests = {
            '0.7.2/001': 'bb533144fb66d283df8c4489b59634b7b6b2ac7d4e61847ad69103e0b979a0a4',
            '0.7.2/002': '81807ea1151ed09c553718566bc200aa5a32f8e237c660fb6412838049775762',
        }
        with closing(version.connect(self.runtime)) as db, db:
            db.execute("UPDATE state SET applied='0.7.2'")
            for key, digest in digests.items():
                current = scripts[key].relative_path
                self.write(current, (version.ROOT / current).read_text())
                old = current.replace('.version_patch/', '.version_autofix/')
                db.execute("""INSERT INTO attempts
                    (id, patch, path, digest, commit_hash, host, attempt, started, status, log_path)
                    VALUES (?, ?, ?, ?, 'legacy', 'test', 1, 0, 'success', '')""", (key, key, old, digest))
        self.assertTrue(version.status(self.root, self.runtime)['ready'])

    def test_manual_upgrade_runs_when_fetch_disabled_and_defers_on_code_lock(self):
        self.git_worker(self.control())
        self.release('0.7.2')
        self.commit()
        with version.code_lease(self.scheduling):
            self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'deferred')
        self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'ready')
        self.assertEqual(git_update._channel_update(self.root, {'enabled': False})['status'], 'disabled')

    def test_worklist_keeps_queue_until_patch_retry_verifies(self):
        from src.proj.util.script.script_lock import ScriptLock

        self.release('0.7.2')
        self.patch_file('0.7.2', '001_wait', """
import time

def apply(ctx):
    (ctx.root.parent / 'started').touch()
    deadline = time.monotonic() + 15
    while not (ctx.root.parent / 'finish').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('test did not release patch')
        time.sleep(.02)

def verify(ctx):
    return (ctx.root.parent / 'valid').exists()
""")
        self.commit()
        store = RunStore(self.scheduling / 'runs.sqlite')
        request = dict(id='queued-model', task='model', version='model-v1', owner=idle.OWNER, phase='queued')
        store.put(request)
        self.enterContext(patch.object(idle, 'runtime_dir', return_value=self.scheduling))
        self.enterContext(patch.object(version, 'business_lease', partial(version.business_lease, root=self.root)))
        self.enterContext(patch.object(idle, 'RunStore', return_value=store))
        self.enterContext(patch.object(ScriptLock, 'LOCK_DIR', self.scheduling / 'script_lock'))
        self.enterContext(patch.object(idle, 'installed_config', return_value={'watchdog': {'jobs': {
            'idle_worklist': {'enabled': True}}}}))
        self.enterContext(patch.object(idle, 'idle_reason', return_value=None))
        self.enterContext(patch.object(idle, 'candidate', return_value=request))
        self.enterContext(patch('src.automation.scheduling.maintenance_gate.permit', return_value=nullcontext()))
        launch = self.enterContext(patch.object(idle, '_launch', return_value=0))

        def assert_waiting(reason):
            self.assertIn(reason, idle.dispatch({'enabled': True}, store)['idle_worklist'])
            self.assertEqual(idle.worker(), 0)
            self.assertEqual(store.owned(idle.OWNER), [request])
            launch.assert_not_called()

        assert_waiting('migration pending')
        with ThreadPoolExecutor(max_workers=1) as pool:
            attempt = pool.submit(self.apply)
            try:
                deadline = time.monotonic() + 10
                while not (self.root.parent / 'started').exists():
                    self.assertLess(time.monotonic(), deadline, 'patch child did not start')
                    self.assertFalse(attempt.done())
                    time.sleep(.02)
                assert_waiting('Code lease occupied')
            finally:
                (self.root.parent / 'finish').touch()
            self.assertEqual(attempt.result(timeout=10)['status'], 'blocked')
        assert_waiting('migration pending')
        (self.root.parent / 'valid').touch()
        self.assertEqual(self.apply(retry='0.7.2/001')['status'], 'ready')
        self.assertEqual(idle.worker(), 0)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0]['id'], request['id'])

    def test_worklist_waits_for_first_ledger_on_legacy_install(self):
        self.use_fresh_runtime()
        with (patch.object(idle, 'runtime_dir', return_value=self.scheduling),
              patch.object(idle, 'installed_config', return_value={'scheduler_version': 3}),
              patch.object(idle, 'RunStore') as store):
            self.assertIn('migration check failed', idle.dispatch({'enabled': True})['idle_worklist'])
            self.assertEqual(idle.worker(), 0)
            store.assert_not_called()
        self.assertFalse(version.ledger_path(self.runtime).exists())

    def test_calendar_and_autofix_wait_for_verified_data(self):
        controls = self.control()
        runs = RunStore(self.scheduling / 'runs.sqlite')
        tasks = {'daily_update': 'data', 'weekly_update': 'data', 'rcquant_sec_backfill': 'independent'}
        config = {'timezone': 'UTC', 'tasks': {name: {'enabled': True, 'lock_group': name, 'channel': group}
                                              for name, group in tasks.items()}}
        self.enterContext(patch.object(version, 'business_lease', partial(version.business_lease, root=self.root)))
        for module in (calendar_lanes, autofix, runner):
            self.enterContext(patch.object(module, 'runtime_dir', return_value=self.scheduling))
        for module in (calendar_lanes, autofix):
            self.enterContext(patch.object(module, 'state_dir', return_value=controls.directory))
        self.enterContext(patch.object(calendar_lanes, 'require_installed_config', return_value=config))
        self.enterContext(patch.object(calendar_lanes, 'RunStore', return_value=runs))
        self.enterContext(patch.object(channels, 'enabled', return_value=True))
        self.enterContext(patch('src.automation.scheduling.maintenance_gate.configured', return_value=True))
        self.enterContext(patch('src.automation.scheduling.maintenance_gate.permit', return_value=nullcontext()))
        self.enterContext(patch('src.automation.scheduling.runtime.process_cgroup', return_value='/learndl-autofix.service'))
        self.enterContext(patch.object(runner, 'installed_config', return_value=None))
        launch = self.enterContext(patch.object(runner, '_run', return_value=0))
        controls.register([{'id': 'test-repair', 'script': 'scripts/test.py'}], 'fixture')
        self.enterContext(patch.object(autofix, 'discover', return_value=True))
        repair = self.enterContext(patch.object(autofix, 'execute',
                                  side_effect=lambda store, root, row: store.update(row['id'], phase='success')))
        legacy_script = self.enterContext(patch.object(runner.subprocess, 'call', return_value=0))

        self.release('0.7.2')
        self.patch_file('0.7.2', '001_check', "def apply(ctx): pass\ndef verify(ctx): return (ctx.root.parent / 'valid').exists()")
        self.commit()
        with version.code_lease(self.scheduling), patch.dict(os.environ, {'LEARNDL_SCHEDULE_SLOT': '123'}):
            for task in tasks:
                runner.run(task)
        queued = controls.scheduled()
        self.assertEqual(len(queued), 3)

        def try_workers():
            for task in tasks:
                self.assertEqual(calendar_lanes.worker(task), 0)
            with patch.object(sys, 'platform', 'linux'):
                self.assertEqual(autofix.worker(), 0)

        def assert_waiting():
            try_workers()
            runner.run_deferred(controls)
            self.assertEqual(runner.legacy('daily_update', []), 75)
            self.assertEqual(controls.scheduled(), queued)
            launch.assert_not_called()
            repair.assert_not_called()
            legacy_script.assert_not_called()

        assert_waiting()
        with version.code_lease(self.scheduling):
            assert_waiting()
        self.assertEqual(self.apply()['status'], 'blocked')
        assert_waiting()
        (self.root.parent / 'valid').touch()
        self.assertEqual(self.apply(retry='0.7.2/001')['status'], 'ready')
        try_workers()
        self.assertEqual([call.args[0] for call in launch.call_args_list], list(tasks))
        self.assertEqual(controls.scheduled(), [])
        repair.assert_called_once()
        self.assertEqual(runner.legacy('daily_update', []), 0)
        legacy_script.assert_called_once()

    def test_calendar_script_holds_patch_lock_across_exec_after_runner_exits(self):
        from src.proj.util.script.script_lock import ScriptLockMultiple

        script = self.write('fixture.py', """
from pathlib import Path
import sys

Path(__file__).with_suffix('.ready').touch()
sys.stdin.buffer.read()
""")
        config = {'tasks': {'daily_update': {'enabled': True, 'entrypoint': 'daily_update', 'args': [],
                                             'lock_group': 'daily_update', 'overlap': 'skip', 'timeout_seconds': 60}}}
        store = RunStore(self.scheduling / 'runs.sqlite')
        self.enterContext(patch.object(runner, 'RunStore', return_value=store))
        self.enterContext(patch.dict(runner.ENTRYPOINTS, daily_update=str(script)))
        self.enterContext(patch.object(ScriptLockMultiple, 'LOCK_DIR', self.scheduling / 'script-locks'))
        popen = subprocess.Popen
        children = []

        def start_child(command, **kwargs):
            child = popen(command, **kwargs, stdin=subprocess.PIPE)
            children.append(child)
            return Mock(pid=child.pid, wait=Mock(side_effect=SystemExit))

        try:
            with (self.assertRaises(SystemExit),
                  version.business_lease(self.scheduling, root=self.root) as fd,
                  patch.object(runner.subprocess, 'Popen', side_effect=start_child)):
                runner._run('daily_update', code_fd=fd, config_snapshot=config)
            child, = children
            deadline = time.monotonic() + 10
            while not script.with_suffix('.ready').exists():
                self.assertIsNone(child.poll())
                self.assertLess(time.monotonic(), deadline, 'calendar script did not start')
                time.sleep(.02)
            with self.assertRaises(version.CodeBusy), version.code_lease(self.scheduling):
                self.fail('Patch entered while calendar script was alive')
        finally:
            for child in children:
                child.communicate(timeout=10)
        with version.code_lease(self.scheduling):
            self.assertEqual(children[0].returncode, 0)

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

    def test_blocked_worker_fetches_and_retries_once_per_new_commit(self):
        store = self.control()
        self.git_worker(store)
        origin = self.root.parent / 'origin'
        self.git('clone', str(self.root), str(origin))
        self.git('remote', 'add', 'origin', str(origin))
        (origin / version.VERSION_FILE).write_text("__version__ = '0.7.2'\n")
        location = origin / version.PATCH_DIRECTORY / '0.7.2' / '001_patch.py'
        location.parent.mkdir(parents=True)
        location.write_text('def apply(ctx): pass\ndef verify(ctx): return False\n')

        def publish(message):
            self.git('-C', str(origin), 'add', '.')
            self.git('-C', str(origin), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', message)

        def worker(expected):
            result = git_update._channel_update(self.root, {'enabled': True})
            self.assertEqual(result['status'], expected, result)
            return version.status(self.root, self.runtime)['records']

        with patch.object(git_update, 'queue_update_email', return_value='notice'):
            publish('broken release')
            self.assertEqual(worker('updated'), [])
            attempts = worker('blocked')
            self.assertEqual(len(attempts), 1)
            self.assertEqual(worker('blocked'), attempts)
            self.assertEqual(priority.read(store)['phase'], 'blocked')

            (origin / 'fix-notes.txt').write_text('new revision, still broken')
            publish('first fix')
            result = git_update._channel_update(
                self.root, {'enabled': True, 'retry_failed_patch_on_update': False})
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(version.status(self.root, self.runtime)['records'], attempts)

            self.assertEqual(worker('updated'), attempts)
            attempts = worker('blocked')
            self.assertEqual([row['status'] for row in attempts], ['failed', 'failed'])
            self.assertNotEqual(attempts[0]['commit_hash'], attempts[1]['commit_hash'])
            self.assertEqual(worker('blocked'), attempts)

            location.write_text('def apply(ctx): pass\ndef verify(ctx): return True\n')
            publish('working fix')
            self.assertEqual(worker('updated'), attempts)
            attempts = worker('verified')
            self.assertEqual([row['status'] for row in attempts], ['failed', 'failed', 'success'])
        self.assertEqual(priority.read(store)['phase'], 'idle')
        self.assertTrue(version.status(self.root, self.runtime)['ready'])

    def test_manual_fast_forward_recovers_on_watchdog_tick(self):
        store = self.control()
        self.git_worker(store)
        self.release('0.7.2')
        path = self.patch_file('0.7.2', '001_patch', 'def apply(ctx): pass\ndef verify(ctx): return False\n')
        self.commit()
        original = self.git('rev-parse', 'HEAD').decode().strip()
        priority.write(store, {'phase': 'checkout', 'last_completed': 0, 'target': original})
        self.assertEqual(git_update._channel_update(self.root, {})['status'], 'blocked')

        origin = self.root.parent / 'origin'
        self.git('clone', str(self.root), str(origin))
        self.git('remote', 'add', 'origin', str(origin))
        target = path.relative_to(self.root)
        (origin / target).write_text('def apply(ctx): pass\ndef verify(ctx): return True\n')
        self.git('-C', str(origin), 'add', str(target))
        self.git('-C', str(origin), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fix')
        self.git('pull', '--ff-only', 'origin', 'master')
        # A fresh worker verifies the new checkout against the failed attempt.
        result = git_update._channel_update(self.root, {'enabled': True})
        self.assertEqual(result['status'], 'verified', result)
        snapshot = version.status(self.root, self.runtime)
        self.assertEqual([row['status'] for row in snapshot['records']], ['failed', 'success'])
        self.assertTrue(snapshot['ready'])

    def test_bootstrap_holds_new_business_before_import_after_manual_pull(self):
        control = self.control().directory
        (control / 'worker-settings.json').write_text(json.dumps({'root': str(self.root), 'runtime': str(self.scheduling)}))
        (self.scheduling / 'installed.json').write_text('{"scheduler_version": 3, "version_patch": true}')
        self.release('0.7.2')
        self.commit()
        self.addCleanup(os.chdir, Path.cwd())
        with (patch.object(worker_bootstrap, '__file__', str(control / 'worker_bootstrap.py')),
              patch.object(sys, 'argv', ['bootstrap', 'lane', 'daily_update']),
              patch.object(sys, 'path', sys.path.copy()),
              patch.object(worker_bootstrap.runpy, 'run_module') as load):
            worker_bootstrap.main()
            load.assert_not_called()
            with patch.object(sys, 'argv', ['bootstrap', 'watchdog']):
                with version.code_lease(self.scheduling):
                    worker_bootstrap.main()
                    load.assert_not_called()
                worker_bootstrap.main()
                load.assert_called_once_with('src.automation.watchdog', run_name='__main__')
                load.reset_mock()
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
            self.assertFalse(worker_bootstrap.wait_for_code(control, 'watchdog'))
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
