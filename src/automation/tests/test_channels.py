"""Real file-lock admission, expiry, independent workers and version-3 policy."""
import os
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo

import portalocker

from src.automation.scheduling import channels, calendar_lanes, priority, maintenance_gate, runner
from src.automation.scheduling.autofix_store import initialize, Store
from src.automation.scheduling.runtime import RunStore


class ChannelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base/'project'
        self.root.mkdir()
        self.control = self.base/'control'
        initialize(self.control, self.root)
        self.store = Store(self.control)
        priority.initialize(self.store)
        channels.initialize(self.store)
        self.config = {'scheduler_version': 3, 'timezone': 'Asia/Shanghai', 'tasks': {
            name: {'channel': channel, 'lock_group': name, 'enabled': True,
                   'window': 'rcquant_evening' if channel == 'independent' else None}
            for name, channel in [('daily_update', 'data'), ('weekly_update', 'data'),
                                  ('rcquant_sec_backfill', 'independent')]},
            'watchdog': {'jobs': {'git_auto_update': {'enabled': True, 'interval_seconds': 60}}}}
        for module in (channels, priority, calendar_lanes):
            self.enterContext(patch.object(module, 'installed_config', return_value=self.config))
        self.enterContext(patch('src.automation.scheduling.install_request.reconcile', return_value={}))
        self.enterContext(patch('src.automation.scheduling.auxiliary.control_error'))
        self.enterContext(patch.object(calendar_lanes, 'state_dir', return_value=self.control))
        self.runs = RunStore(self.base/'runs.sqlite')
        self.enterContext(patch.object(calendar_lanes, 'RunStore', return_value=self.runs))

    def permit(self, kind):
        return maintenance_gate.permit(task_class=kind, directory=self.control, blocking=False)

    def test_due_check_does_not_block_business(self):
        self.assertTrue(priority.git_due(self.store))
        for kind in ('data', 'research', 'independent', 'manual', 'autofix:fix'):
            with self.permit(kind):
                pass

    def test_blocking_scope_and_physical_existing_task(self):
        with channels.leases(self.store, 'research'):
            self.assertTrue(channels.reserve(self.store, 'autofix:fix', ['research', 'data']))
            for kind in ('data', 'research'):
                with self.assertRaises(maintenance_gate.MaintenanceBusy):
                    with self.permit(kind):
                        pass
            for kind in ('independent', 'manual', 'system'):
                with self.permit(kind):
                    pass
            with self.assertRaises(maintenance_gate.MaintenanceBusy):
                with self.permit('autofix:fix'):
                    pass
        with self.permit('autofix:fix'):
            channels.running(self.store, 'autofix:fix')
        channels.release(self.store, 'autofix:fix')
        with self.permit('data'):
            pass

    def test_one_hour_timeout_and_shared_cooldown(self):
        start = time.time()
        self.assertTrue(channels.reserve(self.store, 'autofix:first', ['research'], start))
        self.assertTrue(channels.reserve(self.store, 'autofix:first', ['research'], start+3000))
        self.assertEqual(channels.read(self.store)['requested_at'], start)
        self.assertFalse(channels.reserve(self.store, 'git', channels.GROUPS, start+3600))
        self.assertFalse(channels.reserve(self.store, 'autofix:second', ['data'], start+4499))
        channels.release(self.store, 'autofix:first')
        self.assertEqual(channels.read(self.store)['phase'], 'cooldown')
        self.assertTrue(channels.reserve(self.store, 'git', channels.GROUPS, start+4500))

    def test_running_maintenance_not_expired_by_wait_deadline(self):
        channels.reserve(self.store, 'autofix:fix', ['data'])
        channels.running(self.store, 'autofix:fix')
        self.assertFalse(channels.reserve(self.store, 'git', channels.GROUPS, time.time()+9999))
        self.assertEqual(channels.read(self.store)['phase'], 'running')

    def test_git_preempts_waiting_autofix_without_extending_deadline(self):
        start = time.time()
        channels.reserve(self.store, 'autofix:fix', ['research'], start)
        self.assertTrue(channels.reserve(self.store, 'git', channels.GROUPS, start + 60))
        state = channels.read(self.store)
        self.assertEqual(state['deadline'], start + 3600)
        self.assertEqual(state['preempted_owner'], 'autofix:fix')

    def test_watchdog_can_reconcile_while_git_drains_but_not_during_checkout(self):
        from src.automation.scheduling import worker_bootstrap
        (self.control/'worker-settings.json').write_text(json.dumps({'runtime': str(self.base)}))
        (self.base/'installed.json').write_text(json.dumps(self.config))
        channels.reserve(self.store, 'git', channels.GROUPS)
        self.assertTrue(worker_bootstrap.wait_for_code(self.control, 'lane'))
        self.assertFalse(worker_bootstrap.wait_for_code(self.control, 'watchdog'))
        priority.write(self.store, {'phase': 'checkout', 'target': 'new', 'last_completed': 0})
        self.assertTrue(worker_bootstrap.wait_for_code(self.control, 'watchdog'))

    def test_all_code_users_block_git_including_independent(self):
        with self.permit('independent'):
            with (self.control/'gate.lock').open('r+') as stream:
                with self.assertRaises(portalocker.LockException):
                    portalocker.lock(stream, portalocker.LOCK_EX | portalocker.LOCK_NB)
        channels.reserve(self.store, 'git', channels.GROUPS)
        with self.assertRaises(maintenance_gate.MaintenanceBusy):
            with self.permit('independent'):
                pass

    def test_calendar_registration_dedup_and_overlap(self):
        self.runs.put({'id': 'live', 'phase': 'starting', 'lock_group': 'daily_update'})
        with patch.dict(os.environ, {'LEARNDL_SCHEDULE_SLOT': '100'}):
            calendar_lanes.register('daily_update')
            calendar_lanes.register('daily_update')
            calendar_lanes.register('weekly_update')
        self.assertEqual(self.store.scheduled(), [('weekly_update:100', 'weekly_update')])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT state FROM scheduled WHERE task='daily_update'").fetchone()[0], 'skipped_overlap')

    def test_waiting_requests_coalesce_without_duplicate_execution(self):
        for slot in ('100', '101'):
            with patch.dict(os.environ, {'LEARNDL_SCHEDULE_SLOT': slot}):
                calendar_lanes.register('weekly_update')
        self.assertEqual(self.store.scheduled(), [('weekly_update:101', 'weekly_update')])
        with patch.object(calendar_lanes, 'eligibility', return_value=('ready', '')), \
             patch.object(maintenance_gate, 'state_dir', return_value=self.control), \
             patch.object(runner, '_run', return_value=0) as run:
            calendar_lanes.worker('weekly_update')
            calendar_lanes.worker('weekly_update')
        run.assert_called_once_with('weekly_update', request_id='weekly_update:101')

    def test_rcquant_window_dependency_and_expiry(self):
        stamp = lambda date: datetime.fromisoformat(date).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        key = f"rcquant_sec_backfill:{int(stamp('2026-10-30T21:01')//60)}"
        with patch('src.data.download.other_source.rcquant.bar_min.daily_update_succeeded_today', return_value=False) as daily:
            self.assertEqual(calendar_lanes.eligibility('rcquant_sec_backfill', key, self.config, stamp('2026-10-30T21:01'))[0], 'waiting')
            daily.return_value = True
            self.assertEqual(calendar_lanes.eligibility('rcquant_sec_backfill', key, self.config, stamp('2026-10-30T22:01'))[0], 'ready')
            self.assertEqual(calendar_lanes.eligibility('rcquant_sec_backfill', key, self.config, stamp('2026-10-31T07:24'))[0], 'expired')

    def test_missing_channel_state_fails_closed(self):
        with self.store.connect() as db:
            db.execute('DELETE FROM metadata WHERE key=?', (channels.KEY,))
        with self.assertRaisesRegex(RuntimeError, 'Channel state missing'):
            with self.permit('data'):
                pass

    def test_fetch_up_to_date_does_not_reserve_or_acquire_exclusive(self):
        from src.automation.scheduling import git_update
        def git(root, args, deadline, **kwargs):
            return 'same' if args[0] == 'rev-parse' else ''
        with patch.object(git_update, 'runtime_dir', return_value=self.base), \
             patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.control), \
             patch.object(git_update, 'repository_blocker', return_value=None), \
             patch.object(git_update, '_git', side_effect=git), \
             patch.object(maintenance_gate, 'permit', side_effect=AssertionError('must not reserve')):
            result = git_update._channel_update(self.root, {'enabled': True})
        self.assertEqual(result['status'], 'up_to_date')
        self.assertEqual(channels.read(self.store)['phase'], 'idle')

    def test_available_git_target_waits_for_real_code_reader(self):
        from src.automation.scheduling import git_update
        def git(root, args, deadline, **kwargs):
            if args[:2] == ['rev-parse', 'HEAD'] or args[0] == 'merge-base':
                return 'old'
            return 'new'
        with patch.object(git_update, 'runtime_dir', return_value=self.base), \
             patch('src.automation.scheduling.autofix_store.state_dir', return_value=self.control), \
             patch.object(maintenance_gate, 'state_dir', return_value=self.control), \
             patch.object(git_update, 'repository_blocker', return_value=None), \
             patch.object(git_update, '_git', side_effect=git), \
             patch.object(git_update, '_admitted_update') as update, \
             (self.control/'gate.lock').open('r+') as stream:
            portalocker.lock(stream, portalocker.LOCK_SH)
            result = git_update._channel_update(self.root, {'enabled': True})
            update.assert_not_called()
        self.assertEqual(result['status'], 'deferred')
        self.assertEqual(channels.read(self.store)['phase'], 'waiting')

    def test_scoped_autofix_does_not_wait_for_independent_channel(self):
        with channels.leases(self.store, 'independent'):
            channels.reserve(self.store, 'autofix:repair', ['research', 'data'])
            with self.permit('autofix:repair'):
                pass

    def test_inherited_lane_descriptor_keeps_research_busy(self):
        import subprocess
        import sys
        with self.permit('research'):
            child = subprocess.Popen([sys.executable, '-c', 'import sys; print("ready",flush=True); sys.stdin.readline()'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     pass_fds=maintenance_gate.pass_fds(), text=True)
            self.assertEqual(child.stdout.readline().strip(), 'ready')
        try:
            channels.reserve(self.store, 'autofix:repair', ['research'])
            with self.assertRaises(maintenance_gate.MaintenanceBusy):
                with self.permit('autofix:repair'):
                    pass
        finally:
            child.communicate('done\n', timeout=10)
        with self.permit('autofix:repair'):
            pass
