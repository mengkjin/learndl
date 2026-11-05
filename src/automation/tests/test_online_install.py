"""Online publication, real lock contention, and strictly scoped force termination."""
from __future__ import annotations

__all__ = [
    'OnlineInstallTest',
]
import copy
import json
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import portalocker

from src.automation.scheduling import (install_impact as impact, install_flow as flow,
    force_install as force, install_request as requests, publication, channels, priority,
    maintenance_gate as gate, recovery)
from src.automation.scheduling.autofix_store import Store, initialize
from src.automation.scheduling.runtime import RunStore


class OnlineInstallTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.control = self.root / 'control'
        self.directory = self.root / 'runtime'
        self.directory.mkdir()
        initialize(self.control, self.root / 'project')
        self.store = Store(self.control)
        channels.initialize(self.store)
        priority.initialize(self.store)
        self.tasks: dict[str, dict[str, bool | str | int]] = {
            'daily': dict(enabled=True, channel='data', lock_group='daily', timeout_seconds=100, priority=0),
        }
        self.config = dict(
            scheduler_version=3, timezone='UTC', tasks=self.tasks,
            schedules={'daily': dict(task='daily', every_minutes=1)},
            watchdog={'units': [], 'tick_seconds': 60, 'jobs': {'task_timeouts': {'enabled': True}}},
        )
        self.plan_tasks = copy.deepcopy(self.tasks)
        self.plan_schedules: dict[str, dict[str, str | int | list[str]]] = {
            'daily': dict(task='daily', every_minutes=1),
        }
        self.plan_timeout_job = {'enabled': True}
        self.plan = dict(
            config=dict(self.config, tasks=self.plan_tasks, schedules=self.plan_schedules,
                        watchdog={'units': [], 'tick_seconds': 60,
                                  'jobs': {'task_timeouts': self.plan_timeout_job}}),
            units={'learndl-schedule-daily.timer': 'timer'},
            impact={'mode': 'config', 'groups': []},
        )
        (self.directory / 'installed.json').write_text(json.dumps(self.config))
        (self.directory / 'manifest.json').write_text(json.dumps(
            dict(install_protocol=impact.PROTOCOL, sources=impact.sources(), units=self.plan['units'])))
        for module in (channels, priority):
            self.enterContext(patch.object(module, 'installed_config', return_value=self.config))
        for module in (publication, flow, force):
            self.enterContext(patch.object(module, 'state_dir', return_value=self.control))
        self.enterContext(patch.object(flow, 'runtime_dir', return_value=self.directory))
        self.enterContext(patch('src.automation.scheduling.config.runtime_dir', return_value=self.directory))
        self.enterContext(patch.object(requests, 'directory', return_value=self.root / 'requests'))
        self.runs = RunStore(self.directory / 'runs.sqlite')
        self.enterContext(patch.object(force, 'RunStore', return_value=self.runs))

    def test_classification_defaults_and_scopes(self):
        self.plan_schedules['daily'] = dict(task='daily', times=['21:00'])
        self.assertEqual(impact.classify(self.plan, self.directory)['mode'], 'config')
        self.plan_tasks['daily']['lock_group'] = 'different'
        result = impact.classify(self.plan, self.directory)
        self.assertEqual(result['mode'], 'exclusive')
        self.assertEqual(result['groups'], ['data', 'manual'])
        self.plan_timeout_job['enabled'] = False
        self.assertIsNone(impact.classify(self.plan, self.directory)['groups'])

    def test_compatibility_requires_reviewed_hash_transition(self):
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        key = next(iter(manifest['sources']))
        new = manifest['sources'][key]
        manifest['sources'][key] = 'old'
        (self.directory / 'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(impact.classify(self.plan, self.directory)['mode'], 'exclusive')
        with patch.object(impact, 'COMPATIBLE_SOURCE_UPGRADES', {key: {('old', new)}}):
            self.assertEqual(impact.classify(self.plan, self.directory)['mode'], 'compatible')

    def test_hot_publish_does_not_wait_for_research_lifetime(self):
        with (self.control / 'gate.lock').open('r+') as live:
            portalocker.lock(live, portalocker.LOCK_SH)
            with flow.admission(plan=self.plan):
                publication.freeze(self.store, 'daily:10', 'daily', self.config)
                self.store.enqueue('daily', 10)
                new = copy.deepcopy(self.config)
                tasks = copy.deepcopy(self.tasks)
                tasks['daily']['timeout_seconds'] = 200
                new['tasks'] = tasks
                digest = impact.publish(self.directory, new)
                self.assertTrue((self.directory / 'generations' / (digest + '.json')).exists())
                self.assertEqual(publication.queued_config(self.store, 'daily:10', new), self.config)
            with self.assertRaises(gate.MaintenanceBusy):
                with flow.admission(plan={'impact': {'mode': 'exclusive', 'groups': None}}):
                    self.fail('Exclusive upgrade admitted while task runs')

    def test_scoped_upgrade_only_waits_for_affected_lanes(self):
        plan = {'impact': {'mode': 'exclusive', 'groups': ['data']}}
        with channels.leases(self.store, 'research'):
            with flow.admission(plan=plan):
                pass
        with channels.leases(self.store, 'data'):
            with self.assertRaises(gate.MaintenanceBusy):
                with flow.admission(plan=plan):
                    pass
        with flow.admission(plan=plan):
            pass  # Failed attempt did not retain a partial lease.

    def test_calendar_boundary_and_disable_are_durable(self):
        publication.catch_up(self.directory, self.config, self.plan['units'], 600.5, 720)
        publication.catch_up(self.directory, self.config, self.plan['units'], 600.5, 720)
        self.assertEqual(self.store.scheduled(), [('daily:12', 'daily')])
        self.tasks['daily']['enabled'] = False
        publication.cancel_disabled(self.config)
        self.assertEqual(self.store.scheduled(), [])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM scheduled WHERE state='cancelled'").fetchone()[0], 1)

    def test_waiting_installation_does_not_reserve_business(self):
        row = dict(id='request', phase='waiting', created_at=time.time(), expires_at=time.time()+100,
                   boot_id=None, bundle=str(self.root/'bundle'), groups=None)
        requests.save(row)
        with patch.object(requests, 'reconcile', return_value=row):
            with gate.permit(directory=self.control, blocking=False, task_class='research'):
                pass
            with gate.permit(directory=self.control, blocking=False, task_class='data'):
                pass

    def test_unready_git_does_not_starve_scoped_autofix(self):
        channels.reserve(self.store, 'git', channels.GROUPS)
        channels.reserve(self.store, 'autofix:a', ['data'])
        with (self.control / 'gate.lock').open('r+') as live:
            portalocker.lock(live, portalocker.LOCK_SH)
            with gate.permit(directory=self.control, blocking=False, task_class='autofix:a'):
                channels.running(self.store, 'autofix:a')
        channels.release(self.store, 'autofix:a')
        self.assertEqual(channels.request(self.store, 'git')['phase'], 'waiting')

    def test_force_marks_intent_then_escalates_and_prevents_recovery(self):
        run = dict(id='live', owner='idle_worklist', task='train', phase='running', pid=123,
                   process_created=10, runner_pid=99, uid=1000, boot_id=None, cgroup=None, started_at=1)
        self.runs.put(run)
        self.runs.put(dict(id='retry', phase='queued', parent_run_id='live'))
        target = dict(kind='run', id='live', identity={key: run.get(key) for key in force.IDENTITY})
        approval = dict(id='install', plan={'impact': {'mode': 'exclusive', 'groups': None}}, force_targets=[target])
        requests.save(dict(id='install', phase='waiting'))
        clock = iter([0, 0, 31])
        with patch.object(force, 'targets', return_value=[target]), \
             patch.object(force, 'members', side_effect=[[Mock()], [Mock()], []]), \
             patch.object(force, 'signal_members') as send, patch.object(force.time, 'sleep'), \
             patch.object(force.time, 'monotonic', side_effect=lambda: next(clock)):
            force.drain(approval)
        self.assertEqual([c.args[2] for c in send.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        stopped = self.runs.get('live')
        self.assertEqual(stopped['installation_stop'], 'install')
        self.assertFalse(recovery.recovery_plan(self.runs, stopped, {})['retry'])
        self.assertEqual(self.runs.get('retry')['phase'], 'cancelled')

    def test_force_identity_change_never_signals(self):
        approval = dict(id='install', plan={}, force_targets=[dict(kind='run', id='a', identity={'pid': 1})])
        requests.save(dict(id='install', phase='waiting'))
        with patch.object(force, 'targets', return_value=[dict(kind='run', id='a', identity={'pid': 2})]), \
             patch.object(force, 'signal_members') as send:
            with self.assertRaisesRegex(gate.MaintenanceBusy, 'identities changed'):
                force.drain(approval)
        send.assert_not_called()
        self.assertEqual(requests.current()['phase'], 'waiting')

    def test_natural_completion_is_not_overwritten(self):
        self.runs.put(dict(id='done', phase='complete'))
        force.mark_run(self.runs, 'done', 'install')
        self.assertEqual(self.runs.get('done'), dict(id='done', phase='complete'))

    def test_stale_worker_cannot_erase_stop_intent(self):
        run = dict(id='live', phase='running')
        self.runs.put(run)
        force.mark_run(self.runs, 'live', 'install')
        self.runs.put(dict(run, phase='error'))
        self.assertEqual(self.runs.get('live')['installation_stop'], 'install')
        self.assertIn('term_at', self.runs.get('live'))

    def test_force_survivor_timeout_keeps_intent_and_never_installs(self):
        run = dict(id='live', task='train', phase='running', pid=123, process_created=10)
        self.runs.put(run)
        target = dict(kind='run', id='live', identity={key: run.get(key) for key in force.IDENTITY})
        requests.save(dict(id='install', phase='waiting'))
        with patch.object(force, 'targets', return_value=[target]), patch.object(force, 'members', return_value=[Mock()]), \
             patch.object(force.time, 'monotonic', side_effect=[0, 121]), patch.object(force, 'signal_members') as send:
            with self.assertRaisesRegex(gate.MaintenanceBusy, '120 seconds'):
                force.drain(dict(id='install', plan={}, force_targets=[target]))
        send.assert_not_called()
        self.assertTrue(self.runs.get('live')['suppress_auto_recovery'])

    def test_expired_immediate_executor_enters_repair(self):
        requests.save(dict(id='immediate', phase='installing', groups=[], executor_pid=123,
            executor_created=1, boot_id=None, created_at=time.time(), expires_at=time.time()+100))
        with patch.object(requests, 'boot_id', return_value=None), patch('psutil.Process') as process:
            process.return_value.create_time.return_value = 2
            row = requests.reconcile()
        self.assertEqual(row['phase'], 'repair_required')
        self.assertIsNone(row['groups'])

    def test_interrupted_publication_blocks_bootstrap_but_keeps_notification_worker(self):
        from src.automation.scheduling import worker_bootstrap
        (self.control/'worker-settings.json').write_text(json.dumps({'runtime': str(self.directory)}))
        (self.directory/'transaction.json').write_text(json.dumps({'phase': 'paused'}))
        self.assertTrue(worker_bootstrap.wait_for_code(self.control, 'lane'))
        self.assertFalse(worker_bootstrap.wait_for_code(self.control, 'email_outbox'))

    def test_priority_only_orders_ready_same_lock_requests(self):
        from src.automation.scheduling.calendar_lanes import preferred_request
        self.tasks['urgent'] = dict(self.tasks['daily'], priority=10)
        self.store.enqueue('daily', 10)
        self.store.enqueue('urgent', 11)
        self.assertEqual(preferred_request(self.store, 'daily:10', 'daily', self.config), 'urgent:11')
        self.tasks['urgent']['enabled'] = False
        self.assertEqual(preferred_request(self.store, 'daily:10', 'daily', self.config), 'daily:10')


if __name__ == '__main__':
    unittest.main()
