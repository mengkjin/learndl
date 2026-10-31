"""Autofix contracts; all writes use disposable fixtures, never production data."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.task_monitor.scheduling import autofix, autofix_sandbox, maintenance_gate
from src.api.task_monitor.scheduling.autofix_spec import ManifestError, manifest, validate
from src.api.task_monitor.scheduling.autofix_store import Store, initialize

SCRIPT = '''# description: disposable repair
# autofix: true
# parameters:
#   start:
#     type: int
#     default: 20200101

raise AssertionError("validation must not import me")
@ScriptTool('fixture')
def main(start=20200101, **kwargs):
    pass

if __name__ == '__main__':
    main()
'''


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / 'project'
        (self.root / 'scripts').mkdir(parents=True)
        (self.root / 'scripts/fix.py').write_text(SCRIPT)
        self.control = self.base / 'control'
        initialize(self.control, self.root)
        self.store = Store(self.control)
        self.item = {'id': 'fix-001', 'script': 'scripts/fix.py', 'params': {'start': 20200101}}

    def register(self, items=None):
        self.store.register(items if items is not None else [self.item], 'commit-a')


class SpecTests(Fixture):
    def test_blocking_defaults_and_execution_cap(self):
        self.assertEqual(validate(self.root, self.item)['blocking'], 'none')
        for kind in ('research', 'research_and_data'):
            self.assertEqual(validate(self.root, self.item | {'blocking': kind})['timeout_seconds'], 3600)
            self.assertEqual(validate(self.root, self.item | {'blocking': kind, 'timeout': '5m'})['timeout_seconds'], 300)
        with self.assertRaisesRegex(ValueError, 'blocking'):
            validate(self.root, self.item | {'blocking': 'all'})

    def test_missing_or_non_executing_main_guard_is_rejected(self):
        guard = "if __name__ == '__main__':\n    main()"
        for replacement in ('', "if __name__ == '__main__':\n    pass",
                            "if __name__ != '__main__':\n    main()"):
            with self.subTest(replacement=replacement):
                (self.root / 'scripts/fix.py').write_text(SCRIPT.replace(guard, replacement))
                with self.assertRaisesRegex(ValueError, 'entrypoint'):
                    validate(self.root, self.item)

    def test_static_validation_and_email(self):
        result = validate(self.root, self.item)
        self.assertTrue(result['params']['email'])
        self.assertEqual(result['timeout_seconds'], 43200)
        self.assertIn('--start=20200101', result['args'])
        self.item['params']['email'] = False
        self.assertTrue(validate(self.root, self.item)['params']['email'])

    def test_reject_unknown_reserved_types_and_revision(self):
        for change in ({'revision': 2}, {'params': {'start': '1'}}, {'params': {'source': 'bash'}},
                       {'params': {'email': 'false'}}, {'params': {'wat': True}}, {'timeout': '0h'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(self.root, self.item | change)

    def test_path_symlink_missing_and_optin(self):
        (self.root / 'scripts/link.py').symlink_to(self.root / 'scripts/fix.py')
        for script in ('../fix.py', '/tmp/fix.py', 'scripts/link.py', 'scripts/missing.py'):
            with self.subTest(script=script), self.assertRaises((ValueError, FileNotFoundError)):
                validate(self.root, self.item | {'script': script})
        (self.root / 'scripts/fix.py').write_text(SCRIPT.replace('autofix: true', 'autofix: false'))
        with self.assertRaises(ValueError):
            validate(self.root, self.item)

    def test_duplicate_ids_keys_and_empty(self):
        path = self.root / 'tasks.yaml'
        for text in ('schema_version: 1\nfixes: [{id: a}, {id: a}]', 'schema_version: 1\nschema_version: 1\nfixes: []'):
            path.write_text(text)
            with self.assertRaises(ManifestError):
                manifest(path)
        path.write_text('schema_version: 1\nfixes: []\n')
        self.assertEqual(manifest(path), [])

    def test_offline_preview_does_not_import_project_or_initialize_state(self):
        path = self.root / 'runs/autofix'
        path.mkdir(parents=True)
        (path / 'tasks.yaml').write_text('schema_version: 1\nfixes: []\n')
        env = dict(os.environ, HOME=str(self.base / 'empty-home'), PYTHONDONTWRITEBYTECODE='1')
        code = "import runpy,sys;sys.argv=['autofix','plan','--root',sys.argv[1]];runpy.run_module('src.api.task_monitor.scheduling.autofix',run_name='__main__')"
        result = subprocess.run([sys.executable, '-c', code, str(self.root)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.base / 'empty-home').exists())


class LedgerTests(Fixture):
    def test_legacy_email_policy_change_never_resets_attempt(self):
        from src.api.task_monitor.scheduling.autofix_spec import digest
        legacy = self.item | {'params': {'start': 20200101, 'email': False}}
        self.register([legacy])
        # Reproduce the raw declaration hash saved by the previous implementation.
        with self.store.connect() as db:
            db.execute('UPDATE fixes SET fingerprint=? WHERE id=?',
                       (digest({k: v for k, v in legacy.items() if k != 'id'}), legacy['id']))
        updated = self.item | {'params': {'start': 20200101, 'email': True}}
        self.register([updated])
        record = self.store.get('fix-001')
        self.assertFalse(record['conflict'])
        self.assertEqual(record['phase'], 'queued')
        self.assertTrue(validate(self.root, record['item'])['params']['email'])
        self.assertIsNotNone(self.store.claim('fix-001'))
        self.store.update('fix-001', phase='failed')
        self.register([self.item])  # Omitted email is also mandatory true.
        self.assertFalse(self.store.get('fix-001')['conflict'])
        self.assertIsNone(self.store.claim('fix-001'))
        self.register([updated | {'params': {'start': 20210101, 'email': True}}])
        self.assertTrue(self.store.get('fix-001')['conflict'])

    def test_at_most_once_concurrent_claims(self):
        self.register()
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: self.store.claim(self.item['id']), range(16)))
        self.assertEqual(sum(r is not None for r in results), 1)
        with self.store.connect(readonly=True) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE id='fix-001:starting'").fetchone()[0], 1)
        self.store.update(self.item['id'], phase='failed')
        self.register()
        self.assertIsNone(self.store.claim(self.item['id']))

    def test_frozen_declaration_conflict_and_description_only(self):
        self.register()
        self.register([self.item | {'description': 'new words'}])
        self.assertFalse(self.store.get('fix-001')['conflict'])
        self.register([self.item | {'params': {'start': 20210101}}])
        self.assertTrue(self.store.get('fix-001')['conflict'])
        self.assertIsNone(self.store.claim('fix-001'))

    def test_withdraw_readd_success_and_distinct_retry(self):
        self.register()
        self.register([])
        self.assertEqual(self.store.get('fix-001')['phase'], 'withdrawn')
        self.register()
        self.assertIsNotNone(self.store.claim('fix-001'))
        self.store.update('fix-001', phase='success')
        self.register([])
        self.register([self.item, self.item | {'id': 'fix-002'}])
        self.assertIsNone(self.store.claim('fix-001'))
        self.assertIsNotNone(self.store.claim('fix-002'))

    def test_missing_and_corrupt_ledger_fail_closed(self):
        self.store.path.unlink()
        with self.assertRaises(RuntimeError):
            Store(self.control)
        with self.assertRaises(RuntimeError):
            initialize(self.control, self.root)
        self.assertFalse(self.store.path.exists())

    def test_startup_error_consumes_id_then_other_id_runs(self):
        bad = self.item | {'script': 'scripts/missing.py'}
        self.register([bad, self.item | {'id': 'next'}])
        autofix.execute(self.store, self.root, self.store.get('fix-001'))
        record = self.store.get('fix-001')
        self.assertEqual(record['phase'], 'failed')
        self.assertTrue(record['attempted'])
        self.assertIn('FileNotFoundError', record['error'])
        self.assertIsNone(self.store.claim('fix-001'))
        self.assertIsNotNone(self.store.claim('next'))
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM events').fetchone()[0], 3)

    def test_start_notice_retries_without_restarting_even_with_email_disabled(self):
        self.register([self.item | {'params': {'email': False}}])
        with self.store.connect(readonly=True) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM events').fetchone()[0], 0)
        self.store.claim('fix-001', supervisor_created=0)
        with patch.object(autofix, 'state_dir', return_value=self.control):
            calls = []
            self.assertFalse(autofix.deliver(lambda *a, **k: calls.append(a) or False))
            self.assertTrue(autofix.deliver(lambda *a, **k: calls.append(a) or True))
            self.assertTrue(autofix.deliver(lambda *a, **k: self.fail('duplicate start notice')))
        self.assertEqual(len(calls), 2)
        self.assertIn('starting', calls[0][0])
        self.assertIn('不代表业务脚本已开始', calls[0][1])
        self.assertIn('"email":false', calls[0][1])
        self.assertIsNone(self.store.claim('fix-001'))

    def test_unknown_supervisor_is_terminal(self):
        self.register()
        self.store.claim('fix-001', supervisor_created=0)
        with patch.object(autofix.psutil, 'Process', side_effect=autofix.psutil.NoSuchProcess(999)):
            self.assertFalse(autofix.reconcile(self.store))
        self.assertEqual(self.store.get('fix-001')['phase'], 'unknown')
        self.assertIsNone(self.store.claim('fix-001'))

    def test_mail_retry_does_not_retry_script_and_deduplicates(self):
        self.register()
        self.store.event('fix-001:startup_failed', dict(self.store.get('fix-001'), kind='startup_failed', error='Missing script'))
        self.store.event('fix-001:startup_failed', {'kind': 'startup_failed'})
        with patch.object(autofix, 'state_dir', return_value=self.control):
            calls = []
            self.assertFalse(autofix.deliver(lambda *a, **k: calls.append(a) or False))
            self.assertTrue(autofix.deliver(lambda *a, **k: calls.append(a) or True))
            self.assertTrue(autofix.deliver(lambda *a, **k: self.fail('duplicate mail')))
        self.assertEqual(len(calls), 2)
        self.assertIn('Missing script', calls[0][1])
        self.assertFalse(self.store.get('fix-001')['attempted'])

    def test_schedule_queue_dedup(self):
        self.store.enqueue('daily', 1)
        self.store.enqueue('daily', 1)
        self.store.enqueue('daily', 2)
        self.assertEqual(len(self.store.scheduled()), 2)


class IsolationTests(Fixture):
    def test_old_bubblewrap_reports_path_version_and_upgrade(self):
        from types import SimpleNamespace
        with patch.object(autofix_sandbox.sys, 'platform', 'linux'), \
             patch.object(autofix_sandbox.shutil, 'which', return_value='/usr/bin/bwrap'), \
             patch.object(autofix_sandbox.subprocess, 'run', side_effect=[
                 SimpleNamespace(stdout='--unshare-user', stderr=''),
                 SimpleNamespace(stdout='bubblewrap 0.6.1', stderr='')]):
            with self.assertRaisesRegex(RuntimeError, r'/usr/bin/bwrap.*0.6.1.*0.8.0'):
                autofix_sandbox.preflight()

    def test_backported_capability_is_accepted_without_version_gate(self):
        from types import SimpleNamespace
        with patch.object(autofix_sandbox.sys, 'platform', 'linux'), \
             patch.object(autofix_sandbox.shutil, 'which', return_value='/usr/bin/bwrap'), \
             patch.object(autofix_sandbox.subprocess, 'run', return_value=SimpleNamespace(stdout='', stderr='--disable-userns')) as run:
            self.assertEqual(autofix_sandbox.preflight(), '/usr/bin/bwrap')
            run.assert_called_once()

    def test_same_filesystem_bind_mount_is_rejected(self):
        target = str(self.root / 'external data').replace(' ', r'\040')
        with self.assertRaisesRegex(ValueError, 'Nested mount'):
            autofix_sandbox.check_mounts(self.root, f'42 1 8:1 /outside {target} rw - ext4 /dev/sda rw')

    def test_unreadable_scan_is_not_silently_accepted(self):
        def walk(*args, **kwargs):
            kwargs['onerror'](PermissionError('unreadable directory'))
            return iter(())
        with patch.object(autofix_sandbox.os, 'walk', side_effect=walk), self.assertRaises(PermissionError):
            autofix_sandbox.check_tree(self.root)

    def test_hardlink_alias_is_rejected(self):
        outside = self.base / 'outside'
        outside.write_text('original')
        os.link(outside, self.root / 'linked')
        with self.assertRaises(ValueError):
            autofix_sandbox.check_tree(self.root)

    def test_normal_tree_and_internal_links(self):
        os.link(self.root / 'scripts/fix.py', self.root / 'second.py')
        autofix_sandbox.check_tree(self.root)

    def test_command_is_fail_closed_and_state_not_writable(self):
        args = autofix_sandbox.command(self.root, self.control / 'request.json', ['python', 'script'], executable='/usr/bin/bwrap')
        self.assertIn('--die-with-parent', args)
        self.assertIn('--unshare-pid', args)
        self.assertNotIn('--not-a-security-boundary', args)
        writable = [args[i + 1] for i, word in enumerate(args) if word == '--bind']
        self.assertEqual(writable, [])
        self.assertIn('--ro-bind', args)

    def test_no_linux_fallback(self):
        with patch.object(autofix_sandbox.sys, 'platform', 'darwin'), self.assertRaises(RuntimeError):
            autofix_sandbox.preflight()

    def test_gate_pending_and_priority(self):
        (self.control / 'pending').touch()
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with maintenance_gate.permit(directory=self.control, blocking=False):
                pass
        with maintenance_gate.permit(directory=self.control, blocking=False, priority=True):
            with maintenance_gate.permit(directory=self.control, blocking=False):
                pass

    def test_actual_linux_sandbox(self):
        if sys.platform != 'linux' or os.environ.get('LEARNDL_AUTOFIX_SANDBOX_TESTS') != '1':
            self.skipTest('Linux acceptance: LEARNDL_AUTOFIX_SANDBOX_TESTS=1 and bwrap required')
        outside = self.base / 'outside'
        outside.write_text('original')
        request = self.control / 'request.json'
        request.write_text('{}')
        code = '''import pathlib,subprocess,sys
root,outside,ledger=map(pathlib.Path,sys.argv[1:])
(root/'inside').write_text('allowed')
for path in (outside,ledger):
    for op in (lambda:path.write_text('BAD'), lambda:path.unlink(), lambda:path.rename(root/'stolen')):
        try: op()
        except OSError: pass
        else: raise AssertionError('sandbox escaped')
p=subprocess.run([sys.executable,'-c',"open(__import__('sys').argv[1],'w').write('BAD')",str(outside)])
assert p.returncode != 0
'''
        args = autofix_sandbox.command(self.root, request, [sys.executable, '-c', code, str(self.root), str(outside), str(self.store.path)])
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outside.read_text(), 'original')
        self.store.check()


class ExecutionTests(Fixture):
    def simulate(self, *, started=True, success=True, exit_code=0, email=True, wait=False):
        # A disposable process simulates the sandbox protocol. Production never
        # uses this policy and never falls back to a non-isolated command.
        body = f'''
import json,os,pathlib,sys,time
request=json.loads(pathlib.Path(sys.argv[-1]).read_text())
fd=int(sys.argv[sys.argv.index('--info-fd')+1])
os.write(fd,json.dumps({{'child-pid':os.getpid()}}).encode())
channel=pathlib.Path(request['channel'])
def event(kind,**fields):
    (channel/(kind+'.json')).write_text(json.dumps(dict(id=request['id'],nonce=request['nonce'],kind=kind,**fields)))
if {started!r}:
    event('started',email={email!r})
if {wait!r}:
    time.sleep(5)
else:
    if {started!r}:
        event('finished',success={success!r},status='Success' if {success!r} else 'Error')
    else:
        print('ImportError: example dependency unavailable',file=sys.stderr)
    time.sleep(.1)
sys.exit({exit_code})
'''
        policy = 'import sys\ndef check_tree(root): pass\ndef command(root,request,argv):\n    return [sys.executable,"-c",' + repr(body) + ',"--",str(request)]\n'
        (self.control / 'policy.py').write_text(policy)
        item = self.item | {'params': {'start': 20200101, 'email': email}}
        if wait:
            item['timeout'] = '0.3s'
        self.register([item])
        autofix.execute(self.store, self.root, self.store.get(item['id']))
        with self.store.connect() as db:
            events = [json.loads(row[0]) for row in db.execute('SELECT data FROM events')]
        return self.store.get(item['id']), events

    def test_success_does_not_send_framework_result_mail(self):
        row, events = self.simulate()
        self.assertEqual(row['phase'], 'success')
        self.assertTrue(row['handoff'])
        self.assertEqual([e['kind'] for e in events], ['starting'])
        self.assertIsNone(self.store.claim(row['id']))

    def test_script_failure_even_exit_zero_is_not_success_or_duplicate_mail(self):
        row, events = self.simulate(success=False, exit_code=0)
        self.assertEqual(row['phase'], 'failed')
        self.assertEqual([e['kind'] for e in events], ['starting'])

    def test_legacy_email_false_is_forced_true(self):
        row, events = self.simulate(email=False)
        self.assertEqual(row['phase'], 'success')
        self.assertTrue(row['script_mail']['enabled'])
        self.assertEqual([e['kind'] for e in events], ['starting'])

    def test_import_failure_has_startup_mail_and_error_output(self):
        row, events = self.simulate(started=False, exit_code=1)
        self.assertEqual(row['phase'], 'failed')
        self.assertFalse(row['handoff'])
        self.assertEqual([e['kind'] for e in events], ['starting', 'startup_failed'])
        self.assertIn('ImportError', events[1]['error_tail'])

    def test_timeout_is_one_interruption_event(self):
        row, events = self.simulate(wait=True)
        self.assertEqual(row['phase'], 'timed_out')
        self.assertEqual([e['kind'] for e in events], ['starting', 'interrupted'])
        self.assertIsNone(self.store.claim(row['id']))


class AdmissionTests(Fixture):
    def test_activation_is_protected_from_project_config_changes(self):
        self.store.set_enabled(True)
        with patch.object(maintenance_gate, 'state_dir', return_value=self.control), \
             patch('src.api.task_monitor.scheduling.config.installed_config', return_value=None):
            self.assertTrue(maintenance_gate.configured())
            self.store.path.unlink()
            with self.assertRaises(RuntimeError):
                maintenance_gate.configured()

    def test_inherited_child_keeps_lease_after_parent_context_exits(self):
        child = None
        try:
            with maintenance_gate.permit(directory=self.control):
                child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                         stdin=subprocess.PIPE, pass_fds=maintenance_gate.pass_fds())
            with self.assertRaises(maintenance_gate.MaintenanceBusy):
                with maintenance_gate.permit(directory=self.control, exclusive=True, blocking=False):
                    pass
        finally:
            if child is not None:
                child.communicate(timeout=5)
        with maintenance_gate.permit(directory=self.control, exclusive=True, blocking=False):
            pass

    def test_spawn_registration_crash_keeps_scope_blocked(self):
        self.register()
        self.store.claim('fix-001', supervisor_created=0, scope={'cgroup': '/test'})
        with patch.object(autofix.psutil, 'Process', side_effect=autofix.psutil.NoSuchProcess(999)), \
             patch('src.api.task_monitor.scheduling.runtime.members', return_value=[object()]):
            self.assertTrue(autofix.reconcile(self.store))
        self.assertEqual(self.store.get('fix-001')['phase'], 'starting')
        self.assertIsNone(self.store.claim('fix-001'))

    def test_deferred_calendar_queue_precedes_idle_admission(self):
        self.store.enqueue('daily_update', 1)
        self.assertTrue(maintenance_gate.pending(self.control))
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with maintenance_gate.permit(directory=self.control, blocking=False):
                pass
        with maintenance_gate.permit(directory=self.control, blocking=False, priority=True):
            pass

    def test_surviving_cgroup_keeps_claim_active(self):
        import psutil
        self.register()
        self.store.claim('fix-001', supervisor_created=0)
        self.store.update('fix-001', pid=999999, process_created=0, cgroup='/test', uid=os.getuid())
        with patch.object(autofix.psutil, 'Process', side_effect=psutil.NoSuchProcess(999999)), \
             patch('src.api.task_monitor.scheduling.runtime.members', return_value=[object()]):
            self.assertTrue(autofix.reconcile(self.store))
        self.assertEqual(self.store.get('fix-001')['phase'], 'starting')

    def test_watchdog_timeout_is_persisted_before_signalling(self):
        self.register()
        self.store.claim('fix-001', supervisor_created=0)
        self.store.update('fix-001', pid=999999, process_created=0, effective={'timeout_seconds': 10}, started_at=0)
        with patch('src.api.task_monitor.scheduling.runtime.members', return_value=[]), \
             patch('src.api.task_monitor.scheduling.runtime.signal_members') as send:
            autofix.supervise(self.store, now=20)
            self.assertEqual(self.store.get('fix-001')['term_at'], 20)
            self.assertEqual(send.call_args.args[2], autofix.signal.SIGTERM)
            autofix.supervise(self.store, now=60)
            self.assertEqual(send.call_args.args[2], autofix.signal.SIGKILL)

    def test_other_process_cannot_enter_while_exclusive(self):
        code = '''import sys
from pathlib import Path
from src.api.task_monitor.scheduling.maintenance_gate import permit,MaintenanceBusy
try:
    with permit(directory=Path(sys.argv[1]),blocking=False): pass
except MaintenanceBusy: sys.exit(75)
'''
        with maintenance_gate.permit(directory=self.control, exclusive=True):
            result = subprocess.run([sys.executable, '-c', code, str(self.control)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 75, result.stderr)

    def test_runner_persists_busy_trigger_once(self):
        from src.api.task_monitor.scheduling import runner
        with patch.object(maintenance_gate, 'permit', side_effect=maintenance_gate.MaintenanceBusy()), \
             patch.object(maintenance_gate, 'configured', return_value=True), \
             patch('src.api.task_monitor.scheduling.autofix_store.state_dir', return_value=self.control), \
             patch.object(runner.time, 'time', return_value=120), patch.object(runner, '_run') as launch:
            self.assertEqual(runner.run('daily_update'), 0)
            self.assertEqual(runner.run('daily_update'), 0)
            launch.assert_not_called()
        self.assertEqual(self.store.scheduled(), [('daily_update:2', 'daily_update')])

    def test_ambiguous_active_claim_blocks_regular_tasks_even_if_lock_released(self):
        self.register()
        self.store.claim('fix-001')
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with maintenance_gate.permit(directory=self.control, blocking=False, priority=True):
                pass
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with maintenance_gate.permit(directory=self.control, blocking=False, exclusive=True):
                pass


class ScriptContractTests(unittest.TestCase):
    def test_business_exception_is_propagated_after_script_cleanup(self):
        from unittest.mock import MagicMock
        from src.proj.util.script.autorun import AutoRunTask

        class Context:
            kwargs = {}
            should_forfeit = False
            error_messages = []
            status = 'Starting'
            logger = MagicMock()
            cleaned = False

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.cleaned = True

        task = Context()
        def fail(**kwargs):
            raise ValueError('business error')
        wrapped = AutoRunTask.__call__(task, fail)
        with patch.dict(os.environ, LEARNDL_AUTOFIX_ID='fixture'):
            with self.assertRaisesRegex(RuntimeError, 'did not complete'):
                wrapped()
        self.assertTrue(task.cleaned)
        self.assertEqual(task.status, 'Error')
        # Legacy callers retain their existing result-object behaviour.
        self.assertIs(wrapped(), task)

    def test_script_email_option_and_receipt_are_owned_by_script(self):
        from datetime import datetime
        from types import SimpleNamespace
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.util.web.emailer import Email
        task = SimpleNamespace(email=True, kwargs={}, execution_status='Error', task_name='fix',
                               time_str='now', source='autofix', task_full_name='fix',
                               init_time=datetime.now(), end_time=datetime.now(), status='Error',
                               error_message='business error', exit_message='', exit_files=[], _autofix_primary=True)
        with patch('src.api.task_monitor.scheduling.runtime.termination_requested', return_value=False), \
             patch('src.api.task_monitor.scheduling.recovery.script_result'), \
             patch('src.api.task_monitor.scheduling.runtime.record_script_email'), \
             patch.object(autofix_sandbox, 'script_event') as receipt, \
             patch.object(Email, 'send', return_value=True) as sender:
            AutoRunTask.send_email(task)
            sender.assert_called_once()
            receipt.assert_called_once_with('mail', enabled=True, delivered=True)
            sender.reset_mock()
            task.email = False
            AutoRunTask.send_email(task)
            sender.assert_not_called()


class TypedParameterTests(unittest.TestCase):
    def test_manifest_email_overrides_script_default(self):
        from types import SimpleNamespace
        from src.api.util.backend.recorder import BackendTaskRecorder
        from src.proj.util.script.autorun import AutoRunTask
        for enabled in (True, False):
            with self.subTest(email=enabled), patch.dict(os.environ, LEARNDL_AUTOFIX_ID='fixture'), \
                 patch.object(maintenance_gate, 'sandbox_request', return_value={'params': {'email': enabled}}):
                recorder = BackendTaskRecorder(email=not enabled)
                self.assertIs(recorder.params['email'], True)
                task = SimpleNamespace(kwargs={'email': not enabled} | recorder.params)
                self.assertIs(AutoRunTask.email.fget(task), True)

    def test_autofix_preserves_literal_looking_strings(self):
        from src.api.util.backend.recorder import BackendTaskRecorder
        params = {'label': 'True', 'number': '001', 'empty': '', 'email': False}
        with patch.dict(os.environ, LEARNDL_AUTOFIX_ID='fixture'), \
             patch.object(maintenance_gate, 'sandbox_request', return_value={'params': params}):
            actual = BackendTaskRecorder.parse_kwargs({})
        self.assertEqual(actual, params | {'source': 'autofix', 'email': True})


class WorkerDiagnosticsTests(Fixture):
    def test_waiting_reason_is_durable_without_consuming_id(self):
        from contextlib import nullcontext
        from src.api.task_monitor.scheduling import runtime, git_update
        from src.proj.util.script import code_update_lock
        self.register()
        with patch.object(maintenance_gate, 'configured', return_value=True), \
             patch.object(maintenance_gate, 'permit', return_value=nullcontext()), \
             patch.object(autofix.sys, 'platform', 'linux'), \
             patch.object(autofix, 'state_dir', return_value=self.control), \
             patch.object(autofix, 'discover', return_value=True), \
             patch.object(runtime, 'process_cgroup', return_value='/system.slice/learndl-autofix.service'), \
             patch.object(git_update, 'training_blocker', return_value='managed task is active'), \
             patch.object(code_update_lock, 'training_code_guard', return_value=nullcontext()), \
             patch.object(autofix, 'execute') as execute:
            self.assertEqual(autofix.worker(), 0)
            execute.assert_not_called()
        with self.store.connect(readonly=True) as db:
            report = json.loads(db.execute("SELECT value FROM metadata WHERE key='worker_observation'").fetchone()[0])
        self.assertEqual(report['status'], 'waiting')
        self.assertEqual(report['reason'], 'managed task is active')
        self.assertIn('checked_at', report)
        self.assertFalse(self.store.get('fix-001')['attempted'])


class MailboxSafetyTests(Fixture):
    def test_parent_symlink_cannot_redirect_supervisor_write(self):
        from src.api.task_monitor.scheduling import autofix_mailbox as mailbox
        outside = self.base / 'outside'
        outside.mkdir()
        (self.root / '.local_resources').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(OSError):
            mailbox.create(self.root / '.local_resources/autofix_channels/job')
        self.assertEqual(list(outside.iterdir()), [])

    def test_event_symlink_is_replaced_not_followed(self):
        from src.api.task_monitor.scheduling import autofix_mailbox as mailbox
        channel = self.root / 'channel'
        mailbox.create(channel)
        outside = self.base / 'outside'
        outside.write_text('unchanged')
        (channel / 'terminate.json').symlink_to(outside)
        mailbox.write(channel, 'terminate.json', {'nonce': 'test'})
        self.assertEqual(outside.read_text(), 'unchanged')
        self.assertFalse((channel / 'terminate.json').is_symlink())
        (channel / 'started.json').symlink_to(outside)
        self.assertIsNone(autofix._read_event(channel, 'started', 'test'))

    def test_corrupt_mailbox_does_not_prevent_timeout_signal(self):
        self.register()
        self.store.claim('fix-001', supervisor_created=0)
        self.store.update('fix-001', pid=999999, effective={'timeout_seconds': 1}, started_at=0,
                          channel=str(self.root / 'missing'), nonce='test')
        with patch('src.api.task_monitor.scheduling.runtime.members', return_value=[]), \
             patch('src.api.task_monitor.scheduling.runtime.signal_members') as send:
            autofix.supervise(self.store, now=2)
            send.assert_called_once()


if __name__ == '__main__':
    unittest.main()
