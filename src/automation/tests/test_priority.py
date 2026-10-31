"""Admission races and independent helper contracts, using disposable ledgers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import portalocker

from src.automation.scheduling import priority, maintenance_gate, runner, auxiliary, git_update
from src.automation.scheduling.autofix_store import Store, initialize


class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base / 'project'
        (self.root / 'configs/automation').mkdir(parents=True)
        (self.root / 'configs/automation/autofix.yaml').write_text('schema_version: 1\nfixes: []\n')
        self.directory = self.base / 'control'
        initialize(self.directory, self.root)
        self.store = Store(self.directory)
        self.store.set_enabled(True)
        priority.initialize(self.store)
        priority.write(self.store, {'phase': 'idle', 'last_completed': time.time()})
        self.config = {'scheduler_version': 2, 'watchdog': {'jobs': {
            'git_auto_update': {'enabled': True, 'interval_seconds': 60, 'max_seconds': 45}}}}
        self.enterContext(patch.object(priority, 'installed_config', return_value=self.config))
        self.enterContext(patch('src.automation.scheduling.install_request.reconcile', return_value={}))
        self.enterContext(patch('src.automation.scheduling.autofix.commit', return_value='fixture'))

    def permit(self, task_class, exclusive=False):
        return maintenance_gate.permit(directory=self.directory, blocking=False, task_class=task_class, exclusive=exclusive)

    def test_discovery_blocks_training_without_pending_marker_or_prior_tick(self):
        (self.root / 'configs/automation/autofix.yaml').write_text('schema_version: 1\nfixes:\n- id: repair\n  script: scripts/repair.py\n  params: {}\n')
        self.assertFalse((self.directory / 'pending').exists())
        with self.assertRaisesRegex(maintenance_gate.MaintenanceBusy, 'repair'):
            with self.permit('training'):
                self.fail('training overtook repair')
        self.assertFalse(self.store.get('repair')['attempted'])
        with self.permit('calendar'):
            pass

    def test_calendar_precedes_autofix_and_training(self):
        self.store.enqueue('daily_update', 1)
        for kind in ('autofix', 'training'):
            with self.subTest(kind=kind), self.assertRaisesRegex(maintenance_gate.MaintenanceBusy, 'calendar'):
                with self.permit(kind, exclusive=kind == 'autofix'):
                    pass
        with self.permit('calendar'):
            pass

    def test_due_git_blocks_all_new_data_tasks(self):
        priority.write(self.store, {'phase': 'idle', 'last_completed': 0})
        for kind in ('calendar', 'autofix', 'training'):
            with self.subTest(kind=kind), self.assertRaisesRegex(maintenance_gate.MaintenanceBusy, 'git_auto_update'):
                with self.permit(kind, exclusive=kind == 'autofix'):
                    pass
        with self.permit('git', exclusive=True):
            pass

    def test_existing_training_is_not_preempted_and_next_one_yields(self):
        with self.permit('training'):
            self.store.enqueue('daily_update', 1)
            with self.permit('training'):  # inherited/nested work is the same unit
                pass
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with self.permit('training'):
                pass

    def test_refused_admission_releases_physical_lock(self):
        self.store.enqueue('daily_update', 1)
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with self.permit('autofix', exclusive=True):
                pass
        with self.permit('calendar'):
            pass

    def test_missing_priority_state_fails_closed(self):
        with self.store.connect() as db:
            db.execute('DELETE FROM metadata WHERE key=?', (priority.KEY,))
        with self.assertRaisesRegex(RuntimeError, 'priority state missing'):
            with self.permit('training'):
                pass

    def test_calendar_duplicate_trigger_and_busy_queue(self):
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory), \
             patch.object(maintenance_gate, 'state_dir', return_value=self.directory), \
             patch.object(runner, '_run', return_value=0) as execute, \
             patch.dict(os.environ, {'LEARNDL_SCHEDULE_SLOT': '123'}):
            priority.write(self.store, {'phase': 'idle', 'last_completed': 0})
            runner.run('daily_update')
            execute.assert_not_called()
            self.assertEqual(len(self.store.scheduled()), 1)
            priority.write(self.store, {'phase': 'idle', 'last_completed': time.time()})
            runner.run('daily_update')
            runner.run('daily_update')
            execute.assert_called_once()
            self.assertEqual(self.store.scheduled(), [])

    def test_network_failure_releases_priority_but_checkout_requires_verification(self):
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory), \
             patch.object(git_update, '_update', return_value={'status': 'error', 'reason': 'network'}):
            priority.write(self.store, {'phase': 'draining', 'last_completed': 0})
            result = git_update._admitted_update(self.root, {})
            self.assertEqual(result['status'], 'error')
            self.assertFalse(priority.git_due(self.store))
        priority.write(self.store, {'phase': 'checkout', 'target': 'new', 'last_completed': 0})
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory), \
             patch.object(git_update, 'repository_blocker', return_value=None), \
             patch.object(git_update, '_git', return_value='old'):
            self.assertEqual(git_update._admitted_update(self.root, {})['status'], 'blocked')
            self.assertTrue(priority.git_due(self.store))
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory), \
             patch.object(git_update, 'repository_blocker', return_value=None), \
             patch.object(git_update, '_git', return_value='new'):
            self.assertEqual(git_update._admitted_update(self.root, {})['status'], 'verified')
            self.assertFalse(priority.git_due(self.store))

    def test_unexpected_error_after_checkout_intent_still_ends_dispatch(self):
        def broken_notice(*args):
            priority.write(self.store, {'phase': 'checkout', 'target': 'new', 'last_completed': 0})
            raise RuntimeError('notice database unavailable after merge')
        with patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory), \
             patch.object(git_update, '_update', side_effect=broken_notice):
            result = git_update._admitted_update(self.root, {})
        self.assertTrue(result['checkout_attempted'])
        self.assertEqual(result['status'], 'error')
        self.assertTrue(priority.git_due(self.store))

    def test_missing_control_state_allows_only_mail_failure_reporting(self):
        from src.automation.scheduling.worker_bootstrap import wait_for_code
        with self.store.connect() as db:
            db.execute('DELETE FROM metadata WHERE key=?', (priority.KEY,))
        self.assertFalse(wait_for_code(self.directory, 'email_outbox'))
        for kind in ('training', 'autofix', 'calendar', 'cli_recovery', 'task_monitor_cache'):
            with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, 'Missing scheduling'):
                wait_for_code(self.directory, kind)

    def test_bootstrap_does_not_import_business_code_while_git_owns_code_lock(self):
        from src.automation.scheduling import worker_bootstrap
        runtime = self.base / 'runtime'
        runtime.mkdir()
        (runtime / 'installed.json').write_text(json.dumps(self.config))
        bootstrap = self.directory / 'worker_bootstrap.py'
        bootstrap.write_bytes(Path(worker_bootstrap.__file__).read_bytes())
        (self.directory / 'worker-settings.json').write_text(json.dumps({'root': str(self.root), 'runtime': str(runtime)}))
        package = self.root / 'src/automation/scheduling'
        package.mkdir(parents=True)
        (self.root / 'src/__init__.py').touch()
        marker = self.root / 'imported'
        (package / 'auxiliary.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
        for phase in ('draining', 'checkout', 'blocked'):
            priority.write(self.store, {'phase': phase, 'last_completed': 0})
            subprocess.run([sys.executable, str(bootstrap), 'email_outbox'], check=True, timeout=15)
            self.assertFalse(marker.exists())
        priority.write(self.store, {'phase': 'idle', 'last_completed': time.time()})
        with portalocker.Lock(str(runtime / 'git_update.lock'), timeout=0):
            subprocess.run([sys.executable, str(bootstrap), 'email_outbox'], check=True, timeout=15)
            self.assertFalse(marker.exists())
        subprocess.run([sys.executable, str(bootstrap), 'email_outbox'], check=True, timeout=15)
        self.assertTrue(marker.exists())
        self.config['tasks'] = {'daily_update': {'enabled': True}}
        (runtime / 'installed.json').write_text(json.dumps(self.config))
        # The calendar slot is durable even when no project module can load.
        with portalocker.Lock(str(runtime / 'git_update.lock'), timeout=0):
            subprocess.run([sys.executable, str(bootstrap), 'calendar', 'daily_update'], check=True, timeout=15)
        self.assertEqual([task for _, task in self.store.scheduled()], ['daily_update'])

    def test_watchdog_git_first_and_no_synchronous_helpers_or_email(self):
        from src.automation import watchdog
        names = ['git_auto_update', 'task_lifecycle', 'autofix', 'idle_worklist', *auxiliary.NAMES]
        self.config['watchdog']['units'] = []
        self.config['watchdog']['jobs'] = {name: {'enabled': True, 'interval_seconds': 60} for name in names}
        self.enterContext(patch('src.automation.scheduling.config.installed_config', return_value=self.config))
        self.enterContext(patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.directory))
        self.enterContext(patch.object(priority, 'git_due', return_value=True))
        publish = self.enterContext(patch.object(auxiliary, 'publish'))
        sender = self.enterContext(patch('src.proj.util.web.emailer.Email.send'))
        calls = []
        for name in ('task_lifecycle', 'autofix', 'idle_worklist'):
            self.enterContext(patch.object(watchdog, '_run_' + name,
                side_effect=lambda context, name=name: (calls.append(name), watchdog.JobResult())[1]))
        for name in auxiliary.NAMES:
            self.enterContext(patch.object(watchdog, '_run_' + name, side_effect=AssertionError('helper ran synchronously')))
        for outcome in ({'status': 'up_to_date'}, {'status': 'error', 'reason': 'offline'},
                        {'status': 'updated', 'checkout_attempted': True}):
            calls.clear()
            with patch.object(watchdog, '_run_git_auto_update',
                              side_effect=lambda _: (calls.append('git'), watchdog.JobResult(stats=outcome))[1]):
                watchdog.run_watchdog(task_db=MagicMock(), cache=MagicMock(),
                                      state_path=self.base / (outcome['status'] + '.json'), units=[])
            self.assertEqual(calls[0], 'git')
            self.assertEqual(calls[1:], [] if outcome.get('checkout_attempted') else ['task_lifecycle', 'autofix', 'idle_worklist'])
        self.assertEqual(publish.call_count, 2)
        sender.assert_not_called()

    def test_installer_provisions_independent_bounded_helpers(self):
        from src.automation.scheduling.config import load_config
        from src.automation.scheduling.installer import build_plan, validate_unit
        from src.proj import PATH
        config = load_config(PATH.main / 'configs/automation', 'mengkjin-server')
        plan = build_plan(config, '', PATH.main, Path(sys.executable), config['timezone'])
        self.assertEqual(plan['config']['scheduler_version'], 3)
        for name in auxiliary.NAMES:
            unit = 'learndl-aux-' + name.replace('_', '-') + '.service'
            contents = plan['units'][unit]
            validate_unit(unit, contents)
            self.assertIn('worker_bootstrap.py', contents)
            self.assertNotIn('TimeoutStartSec=infinity', contents)
            self.assertIn(unit.replace('.service', '.timer'), plan['units'])


class HelperQueueTests(unittest.TestCase):
    def test_failed_delivery_survives_snapshot_cleanup_without_duplicate_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(auxiliary, 'runtime_dir', return_value=Path(temp)):
            for target in ('src.automation.scheduling.runtime.deliver_timeout_events',
                           'src.automation.scheduling.install_request.deliver',
                           'src.automation.scheduling.git_update.deliver_update_emails',
                           'src.api.task_monitor.interactive.maintain',
                           'src.automation.cli_recovery.deliver'):
                self.enterContext(patch(target, return_value=True))
            self.enterContext(patch('src.automation.scheduling.runtime.timeout_task_ids', return_value=set()))
            self.enterContext(patch.object(maintenance_gate, 'configured', return_value=False))
            context = MagicMock()
            context.task_db.get_task.return_value = None
            sender = MagicMock(return_value=False)
            auxiliary.publish({'alerts': {'pending_tasks': ['dead'], 'unit_alerted': {}}}, [], False)
            with self.assertRaisesRegex(RuntimeError, 'delivery pending'):
                auxiliary.deliver(context, sender)
            with auxiliary.queue() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0], 0)
            sender.return_value = True
            auxiliary.deliver(context, sender)
            auxiliary.deliver(context, sender)
            self.assertEqual(sender.call_count, 2)
            self.assertFalse(sender.call_args.kwargs['queue_on_failure'])

    def test_script_outbox_error_does_not_skip_scheduler_notifications(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(auxiliary, 'runtime_dir', return_value=Path(temp)), \
             patch.object(priority, 'enabled', return_value=True), \
             patch.object(auxiliary, 'installed_config', return_value={'watchdog': {'jobs': {
                 'email_outbox': {'enabled': True, 'interval_seconds': 60}}}}), \
             patch('src.api.util.backend.task.TaskDatabase'), patch('src.api.task_monitor.output.OutputCache'), \
             patch('src.proj.util.web.emailer.Email.retry_pending', side_effect=OSError('script outbox unavailable')), \
             patch.object(auxiliary, 'deliver') as deliver:
            self.assertEqual(auxiliary.worker('email_outbox'), 1)
            deliver.assert_called_once()
            state = json.loads((Path(temp) / 'auxiliary/email_outbox.json').read_text())
            self.assertIn('script outbox unavailable', state['error'])

    def test_snapshot_cursor_never_reuses_deleted_ids(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(auxiliary, 'runtime_dir', return_value=Path(temp)):
            value = {'alerts': {'pending_tasks': ['one']}}
            auxiliary.publish(value, [], False)
            with auxiliary.queue() as db:
                first = db.execute('SELECT id FROM snapshots').fetchone()[0]
                db.execute('DELETE FROM snapshots')
            auxiliary.publish(value, [], False)
            with auxiliary.queue() as db:
                second = db.execute('SELECT id FROM snapshots').fetchone()[0]
            self.assertGreater(second, first)

    def test_control_errors_are_deduplicated_without_sending_mail(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(auxiliary, 'runtime_dir', return_value=Path(temp)):
            auxiliary.control_error(RuntimeError('missing ledger'))
            auxiliary.control_error(RuntimeError('missing ledger'))
            with auxiliary.queue() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM errors').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
