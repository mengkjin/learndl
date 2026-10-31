"""Installation authorization, immutable privileges and admission without real sudo/systemd."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.automation.scheduling import install_flow as flow
from src.automation.scheduling import install_request as requests
from src.automation.scheduling import install_privileged as privileged
from src.automation.scheduling import maintenance_gate as gate
from src.automation.scheduling.autofix_store import initialize, Store


class InstallationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.control = self.root / 'control'
        initialize(self.control, self.root / 'project')
        self.enterContext(patch.object(requests, 'directory', return_value=self.root / 'requests'))
        self.enterContext(patch.object(requests, 'boot_id', return_value='boot'))
        self.enterContext(patch.object(flow, 'state_dir', return_value=self.control))
        self.enterContext(patch.object(flow, 'runtime_dir', return_value=self.root / 'runtime'))
        self.row = dict(id='a' * 32, phase='waiting', boot_id='boot', created_at=time.time(),
                        expires_at=time.time() + 86400, bundle=str(self.root / 'bundle'), unit='installation.service')

    def test_expiry_reboot_and_mail_retry(self):
        for reason in ('expiry', 'reboot'):
            row = dict(self.row)
            row.update(expires_at=0) if reason == 'expiry' else row.update(boot_id='old')
            requests.save(row)
            self.assertEqual(requests.reconcile()['phase'], 'cancelled')
        sender = Mock(side_effect=[False, True])
        self.assertFalse(requests.deliver(sender))
        self.assertTrue(requests.deliver(sender))
        requests.deliver(sender)
        self.assertEqual(sender.call_count, 2)

    def test_interrupted_transaction_keeps_admission_closed(self):
        requests.save(dict(self.row, phase='installing', boot_id='old'))
        self.assertEqual(requests.reconcile()['phase'], 'repair_required')
        self.assertTrue(requests.pending())
        with self.assertRaisesRegex(RuntimeError, 'repaired'):
            requests.cancel()

    def test_cancel_waiting_and_registration_timeout(self):
        requests.save(self.row)
        self.assertEqual(requests.cancel()['phase'], 'cancelled')
        self.assertFalse(requests.pending())
        requests.save(dict(self.row, phase='registering', created_at=time.time() - 121))
        self.assertEqual(requests.reconcile()['phase'], 'failed')

    def test_installation_blocks_shared_priority_and_exclusive_but_not_owner(self):
        requests.save(self.row)
        for kwargs in ({}, {'priority': True}, {'exclusive': True}):
            with self.assertRaises(gate.MaintenanceBusy):
                with gate.permit(directory=self.control, blocking=False, **kwargs):
                    self.fail('Task admitted during installation')
        with gate.permit(directory=self.control, exclusive=True, blocking=False, installation_id=self.row['id']):
            with gate.permit(directory=self.control):
                pass  # Nested work under the already admitted installer is permitted.
        requests.cancel()
        with gate.permit(directory=self.control, blocking=False):
            pass

    def test_disabled_autofix_still_uses_existing_admission(self):
        self.assertFalse(Store(self.control).enabled())
        requests.save(self.row)
        with patch.object(gate, 'state_dir', return_value=self.control):
            with self.assertRaises(gate.MaintenanceBusy):
                with gate.permit(blocking=False):
                    self.fail('Disabled autofix bypassed installation gate')

    def interactive(self, *, deferred=False, busy=False, accepted=True, failed=False):
        from src.proj import PATH
        plan = dict(units={}, cron='', conflicts={}, global_conflicts=[], config={}, report={})
        snapshot = dict(files={}, cron='', timezone='UTC')
        helper = Path(flow.__file__).with_name('install_privileged.py').resolve()
        snapshot['files'][str(helper)] = flow.hashlib.sha256(helper.read_bytes()).hexdigest()
        patches = [patch.object(flow.sys, 'platform', 'linux'), patch.object(flow.os, 'getuid', return_value=1000),
                   patch.object(flow.install, 'read_cron', return_value=''),
                   patch.object(flow.install, 'command', return_value='UTC'), patch.object(flow, 'load_config', return_value={}),
                   patch.object(flow.install, 'build_plan', return_value=plan), patch.object(flow, 'preview'),
                   patch.object(flow, 'baseline', return_value=snapshot), patch('builtins.input', return_value='y' if accepted else 'n'),
                   patch.object(flow.subprocess, 'run'), patch.object(flow, 'initialize'),
                   patch.object(flow, 'verify'), patch.object(PATH, 'main', self.root),
                   patch.object(flow, 'admission', return_value=nullcontext()),
                   patch.object(flow, 'check_business_tasks', side_effect=gate.MaintenanceBusy('running') if busy else None)]
        for item in patches:
            self.enterContext(item)
        transaction = self.enterContext(patch.object(flow, 'transact', side_effect=ValueError('bad install') if failed else None))
        register = self.enterContext(patch.object(flow, 'register'))
        return transaction, register

    def test_idle_modes_use_same_transaction_no_background(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                transaction, register = self.interactive(deferred=deferred)
                self.assertEqual(flow.run_interactive(deferred=deferred, host='test'), 0)
                transaction.assert_called_once()
                register.assert_not_called()

    def test_busy_manual_exits_and_deferred_registers(self):
        transaction, register = self.interactive(busy=True)
        self.assertEqual(flow.run_interactive(deferred=False, host='test'), 1)
        register.assert_not_called()
        self.assertEqual(flow.run_interactive(deferred=True, host='test'), 0)
        register.assert_called_once()
        transaction.assert_not_called()

    def test_rejected_confirmation_does_not_authorize_or_install(self):
        transaction, register = self.interactive(accepted=False)
        flow.run_interactive(deferred=True, host='test')
        flow.subprocess.run.assert_not_called()
        transaction.assert_not_called()
        register.assert_not_called()

    def test_installation_error_never_queues(self):
        _, register = self.interactive(failed=True)
        with self.assertRaisesRegex(ValueError, 'bad install'):
            flow.run_interactive(deferred=True, host='test')
        register.assert_not_called()

    def test_privilege_broker_rejects_unapproved_commands_and_content(self):
        approval = dict(id='a' * 32, expires_at=time.time() + 100, allowed_units={'learndl-watchdog.timer': ['approved']})
        bad = [dict(argv=['sudo', 'sh', '-c', 'whoami']),
               dict(argv=['sudo', 'systemctl', 'restart', 'ssh.service']),
               dict(argv=['sudo', 'systemctl', 'start', 'learndl-watchdog.service']),
               dict(argv=['sudo', 'install', '-m', '0644', '/tmp/x', '/etc/systemd/system/learndl-watchdog.timer'], content='changed'),
               dict(argv=['sudo', 'mv', '/etc/passwd', '/etc/passwd.learndl-disabled'])]
        for request in bad:
            with self.assertRaises(ValueError):
                privileged.execute(request, approval)
        with patch.object(privileged, 'UNITS', self.root):
            privileged.execute(dict(argv=['sudo', 'install', '-m', '0644', '/ignored', str(self.root / 'learndl-watchdog.timer')],
                                    content='approved'), approval)
        self.assertEqual((self.root / 'learndl-watchdog.timer').read_text(), 'approved')

    def test_background_worker_uses_frozen_plan_then_finishes(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        (bundle / 'approval.json').write_text(json.dumps(dict(id=self.row['id'], plan={})))
        requests.save(self.row)
        with patch.object(flow, 'admission', return_value=nullcontext()), patch.object(flow, 'verify'), \
             patch.object(flow, 'check_business_tasks'), patch.object(flow, 'transact') as transaction, \
             patch.object(requests, 'deliver'):
            self.assertEqual(flow.worker(bundle), 0)
        transaction.assert_called_once()
        self.assertEqual(requests.current()['phase'], 'success')

    def test_background_changed_inputs_cancel_without_installing(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        (bundle / 'approval.json').write_text(json.dumps(dict(id=self.row['id'], plan={})))
        requests.save(self.row)
        with patch.object(flow, 'admission', return_value=nullcontext()), patch.object(flow, 'verify', side_effect=ValueError('changed')), \
             patch.object(flow, 'transact') as transaction, patch.object(requests, 'deliver'):
            flow.worker(bundle)
        transaction.assert_not_called()
        self.assertEqual(requests.current()['phase'], 'cancelled')

    def test_failed_registration_reopens_admission(self):
        approval = {key: self.row[key] for key in ('id', 'unit', 'boot_id', 'created_at', 'expires_at')}
        approval['helper_source'] = 'pass'
        with patch.object(flow.subprocess, 'run', side_effect=RuntimeError('registration failed')):
            with self.assertRaisesRegex(RuntimeError, 'registration failed'):
                flow.register(approval)
        self.assertFalse(requests.pending())

    def test_rollback_uses_saved_snapshot_after_partial_install(self):
        directory = flow.runtime_dir()
        directory.mkdir()
        backup = dict(config={}, cron='old cron', manifest={'units': {}}, desktop_autostart=None)
        (directory / 'rollback.json').write_text(json.dumps(backup))
        def apply(plan, cron, directory):
            (directory / 'transaction.json').write_text(json.dumps({'phase': 'paused'}))
            raise RuntimeError('install failed')
        approval = dict(plan={'conflicts': {}}, baseline={'cron': 'old cron'})
        with patch.object(flow.install, '_apply_plan', side_effect=[None]) as execute:
            def apply_once(plan, cron, directory):
                if execute.call_count == 1:
                    return apply(plan, cron, directory)
                (directory / 'transaction.json').write_text(json.dumps({'phase': 'complete'}))
            execute.side_effect = apply_once
            with patch.object(flow.install, 'read_cron', return_value='new cron'):
                with self.assertRaisesRegex(RuntimeError, 'install failed'):
                    flow.transact(approval)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(execute.call_args.args[0]['cron'], 'old cron')

    def test_authentication_failure_never_queues(self):
        _, register = self.interactive()
        flow.subprocess.run.side_effect = RuntimeError('authentication failed')
        with self.assertRaisesRegex(RuntimeError, 'authentication failed'):
            flow.run_interactive(deferred=True, host='test')
        register.assert_not_called()
        self.assertFalse(requests.pending())

    def test_registration_handshake_does_not_install_in_submitter(self):
        approval = {key: self.row[key] for key in ('id', 'unit', 'boot_id', 'created_at', 'expires_at')}
        approval['helper_source'] = 'pass'
        real_path = Path
        def bundle_path(value):
            return self.root if value == '/run/learndl-install' else real_path(value)
        bundle = self.root / self.row['id']
        bundle.mkdir()
        (bundle / 'ready').write_text('123')
        with patch.object(flow, 'Path', side_effect=bundle_path), patch.object(flow.subprocess, 'run'), \
             patch.object(flow, 'transact') as transaction:
            self.assertEqual(flow.register(approval), self.row['id'])
        self.assertEqual(requests.current()['phase'], 'waiting')
        transaction.assert_not_called()

    def test_real_generated_plan_passes_privileged_registration(self):
        from src.automation.scheduling.config import load_config
        from src.proj import PATH
        config = load_config(PATH.main / 'configs/automation', 'mengkjin-server')
        plan = flow.install.build_plan(config, '', PATH.main, Path(sys.executable), config['timezone'])
        identifier = 'b' * 32
        approval = dict(id=identifier, uid=1000, unit=f'learndl-install-{identifier}.service',
                        allowed_units={name: [contents] for name, contents in plan['units'].items()})
        payload = self.root / 'approval.json'
        payload.write_text(json.dumps(approval))
        staging = self.root / 'privileged-fixture'
        actual_stat = Path.stat
        def stat(path, *args, **kwargs):
            if path == staging:
                return SimpleNamespace(st_uid=0, st_mode=0o40755)
            return actual_stat(path, *args, **kwargs)
        with patch.object(privileged, 'BASE', staging), patch.dict(os.environ, {'SUDO_UID': '1000'}), \
             patch.object(Path, 'stat', stat), \
             patch.object(privileged.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            privileged.register(payload)
        run.assert_called_once()
        saved = json.loads((staging / identifier / 'approval.json').read_text())
        self.assertEqual(saved['allowed_units'], approval['allowed_units'])
        for suffix in ('service', 'timer'):
            for name in ('email-outbox', 'cli-recovery', 'task-monitor-cache'):
                self.assertIn(f'learndl-aux-{name}.{suffix}', saved['allowed_units'])

    def test_privileged_registration_rejects_unknown_helpers_before_mutation(self):
        for name in ('learndl-aux-shell.service', 'learndl-aux-email-outbox.service.extra',
                     '../learndl-aux-email-outbox.timer', 'ssh.service'):
            payload = self.root / 'invalid.json'
            payload.write_text(json.dumps(dict(id='c' * 32, uid=1000,
                unit='learndl-install-' + 'c' * 32 + '.service', allowed_units={name: ['anything']})))
            with patch.dict(os.environ, {'SUDO_UID': '1000'}), patch.object(privileged, 'BASE') as base:
                with self.assertRaisesRegex(ValueError, 'Invalid unit name'):
                    privileged.register(payload)
                base.mkdir.assert_not_called()

    def test_registration_error_reports_stderr_without_dumping_frozen_source(self):
        approval = {key: self.row[key] for key in ('id', 'unit', 'boot_id', 'created_at', 'expires_at')}
        approval['helper_source'] = '# large frozen helper source'
        failure = subprocess.CalledProcessError(1, ['sudo', '-c', approval['helper_source']],
                                               stderr='ValueError: Invalid unit name: fixture.service')
        with patch.object(flow.subprocess, 'run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'Invalid unit name: fixture.service') as caught:
                flow.register(approval)
        self.assertNotIn(approval['helper_source'], str(caught.exception))
        self.assertEqual(requests.current()['phase'], 'failed')

    def test_expired_privileges_and_symlinks_are_rejected(self):
        approval = dict(id='a' * 32, expires_at=0, allowed_units={})
        with self.assertRaisesRegex(ValueError, 'expired'):
            privileged.execute(dict(argv=['sudo', 'systemctl', 'daemon-reload']), approval)
        approval.update(expires_at=time.time() + 100, allowed_units={'learndl-watchdog.timer': ['approved']})
        (self.root / 'learndl-watchdog.timer').symlink_to(self.root / 'target')
        with patch.object(privileged, 'UNITS', self.root):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                privileged.execute(dict(argv=['sudo', 'install', '-m', '0644', '/ignored',
                                              str(self.root / 'learndl-watchdog.timer')], content='approved'), approval)


    def test_parked_legacy_script_does_not_deadlock_installation(self):
        from src.proj import PATH
        proc = Mock(pid=123, info={'cmdline': [str(PATH.main / flow.ENTRYPOINTS['daily_update'])]})
        proc.create_time.return_value = 10
        with patch('src.automation.scheduling.autofix.recorded_task_blocker', return_value=None), \
             patch('src.automation.scheduling.git_update.training_blocker', return_value=None), \
             patch.object(flow.psutil, 'process_iter', return_value=[proc]):
            with self.assertRaises(gate.MaintenanceBusy):
                flow.check_business_tasks({})
            marker = self.control / 'waiting-123.json'
            marker.write_text(json.dumps({'pid': 123, 'created': 10}))
            flow.check_business_tasks({})
            marker.write_text(json.dumps({'pid': 123, 'created': 9}))
            with self.assertRaises(gate.MaintenanceBusy):
                flow.check_business_tasks({})

    def test_calendar_backlog_drains_after_cancel_with_autofix_disabled(self):
        from src.automation.scheduling import runner
        requests.save(self.row)
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.control), \
             patch.object(gate, 'state_dir', return_value=self.control), \
             patch.object(runner, '_run', return_value=0) as execute:
            runner.run('daily_update')
            self.assertTrue(Store(self.control).scheduled())
            execute.assert_not_called()
            requests.cancel()
            runner.run('weekly_update')
            self.assertFalse(Store(self.control).scheduled())
            self.assertEqual([call.args[0] for call in execute.call_args_list], ['daily_update', 'weekly_update'])


if __name__ == '__main__':
    unittest.main()
