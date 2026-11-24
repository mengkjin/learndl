"""Installation drift, authorization boundaries and restricted timer writes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import TypedDict
import unittest
from unittest.mock import Mock, patch

from src.automation.scheduling import install_check as check, retiming_privileged as privileged
from src.automation.scheduling import git_update

TIMER = ('[Unit]\nDescription=Daily\n[Timer]\nOnCalendar=Mon *-*-* 12:00:00 UTC\n'
         'AccuracySec=1s\nPersistent=false\n[Install]\nWantedBy=timers.target\n')
NAME = 'learndl-schedule-daily.timer'


class _Schedule(TypedDict):
    task: str
    days: list[str]
    times: list[str]


class _Config(TypedDict):
    host: str
    timezone: str
    tasks: dict[str, dict[str, bool | str | int]]
    schedules: dict[str, _Schedule]


class _Grant(TypedDict):
    host: str
    config_dir: str
    tasks: list[str]
    task_definitions: dict[str, dict[str, bool | str | int]]
    timezone: str
    timers: dict[str, dict[str, str]]


class InstallationCheckTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.enterContext(patch.object(check, 'runtime_dir', return_value=self.directory))
        self.enterContext(patch.object(git_update, 'runtime_dir', return_value=self.directory))
        self.old = _Config(host='host', timezone='UTC', tasks={'daily': {'enabled': True}},
                           schedules={'daily': {'task': 'daily', 'days': ['mon'], 'times': ['12:00']}})
        self.new = copy.deepcopy(self.old)
        self.new['schedules']['daily']['times'] = ['13:00', '15:00']
        self.location = dict(host='host', config_dir='/configuration')
        self.grant = _Grant(host=self.location['host'], config_dir=self.location['config_dir'], tasks=['daily'],
                            task_definitions=self.old['tasks'], timezone='UTC',
                            timers={NAME: {'envelope': privileged.envelope(TIMER)}})
        self.plan = dict(config=self.new, units={NAME: TIMER.replace('12:00', '13:00')},
                         conflicts={}, global_conflicts=[], impact={'mode': 'config', 'groups': []})
        self.manifest = dict(units={NAME: TIMER}, cron='')
        self.changes = {'installed.json': 'times changed', NAME: '12 -> 13'}
        self.data = (self.location, self.plan, self.old, self.manifest, '', {NAME: TIMER}, self.changes)

    def test_only_times_on_an_existing_enabled_task_are_authorized(self):
        self.assertTrue(check.authorized(self.data, self.grant))
        for field, value in [('enabled', False), ('entrypoint', 'new-code'), ('timeout_seconds', 30)]:
            with self.subTest(field=field):
                modified = copy.deepcopy(self.new)
                modified['tasks']['daily'][field] = value
                self.assertFalse(check.retiming_tasks(self.old, modified))
        self.new['tasks']['new_task'] = dict(enabled=True)
        self.assertFalse(check.authorized(self.data, self.grant))

    def test_mixed_framework_changes_and_actual_file_drift_require_approval(self):
        self.changes['src/automation/scheduling/worker_bootstrap.py'] = 'changed'
        self.assertFalse(check.authorized(self.data, self.grant))
        self.changes.pop('src/automation/scheduling/worker_bootstrap.py')
        self.data[-2][NAME] = 'external edits'
        self.assertFalse(check.authorized(self.data, self.grant))

    def test_grant_cannot_follow_a_changed_task_identity(self):
        self.grant['task_definitions'] = {'daily': dict(enabled=True, entrypoint='other')}
        self.assertFalse(check.authorized(self.data, self.grant))
        self.assertFalse(check.authorized(self.data, None))

    def test_mail_is_durable_deduplicated_and_reappearing_drift_is_new_episode(self):
        with patch.object(check, 'installed_config', return_value=self.old):
            check.notify('authorization required', self.changes, self.location)
            check.notify('authorization required', self.changes, self.location)
            sender = Mock(side_effect=[False, True, True])
            self.assertFalse(git_update.deliver_update_emails(sender, notice_id=None))
            self.assertTrue(git_update.deliver_update_emails(sender))
            self.assertIn('plan --host host --config-dir /configuration --diff', sender.call_args.args[1])
            self.assertIn('defer --host host', sender.call_args.args[1])
            check.clear_episode()
            check.notify('authorization required', self.changes, self.location)
            self.assertTrue(git_update.deliver_update_emails(sender))
            self.assertEqual(sender.call_count, 3)
            self.assertEqual(sender.call_args_list[0].kwargs['delivery_id'], sender.call_args_list[1].kwargs['delivery_id'])
            self.assertNotEqual(sender.call_args_list[1].kwargs['delivery_id'], sender.call_args_list[2].kwargs['delivery_id'])

    def test_pending_patch_blocks_automatic_publication_but_still_notifies(self):
        from contextlib import nullcontext
        with (patch.object(check, 'context', return_value=self.location),
              patch.object(check, 'snapshot', return_value=self.data),
              patch.object(check, 'permission', return_value=self.grant),
              patch.object(check.install_request, 'current', return_value={}),
              patch.object(check.version_patch, 'code_lease', return_value=nullcontext()),
              patch.object(check.version_patch, 'needs_migration', return_value=True),
              patch.object(check.install_flow, 'transact') as transact,
              patch.object(check, 'notify') as notify):
            check.check()
            transact.assert_not_called()
            self.assertIn('version patch', notify.call_args.args[0])

    def test_automatic_transaction_uses_restricted_commands_and_stops_after_failure(self):
        from contextlib import nullcontext
        transact = Mock(side_effect=RuntimeError('publication failed'))
        with (patch.object(check, 'context', return_value=self.location),
              patch.object(check, 'snapshot', return_value=self.data),
              patch.object(check, 'permission', return_value=self.grant),
              patch.object(check.install_request, 'current', return_value={}),
              patch.object(check.version_patch, 'code_lease', side_effect=lambda *a, **k: nullcontext()),
              patch.object(check.version_patch, 'needs_migration', return_value=False),
              patch.object(check.install_flow, 'admission', side_effect=lambda **k: nullcontext()),
              patch.object(check.install_flow, 'transact', transact),
              patch.object(check, 'notify')):
            check.check()
            check.check()
            self.assertEqual(transact.call_count, 1)
            self.assertFalse(check.RETIMING.get())
            (self.directory / 'automatic-retiming-failure.json').unlink()
            transact.side_effect = lambda approval: self.assertTrue(check.RETIMING.get())
            check.check()
            self.assertEqual(transact.call_count, 2)
            self.assertFalse(check.RETIMING.get())

    def test_fresh_git_worker_checks_installation_even_when_fetch_is_disabled(self):
        order = []
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                 patch('src.automation.scheduling.channels.enabled', return_value=True), \
                 patch('src.automation.scheduling.config.installed_config', return_value={
                     'watchdog': {'jobs': {'git_auto_update': {'enabled': enabled}}}}), \
                 patch.object(check, 'check', side_effect=lambda: order.append('check')), \
                 patch.object(git_update, 'check_and_update', side_effect=lambda options: (
                     order.append('update') or {'status': 'up_to_date'})):
                self.assertEqual(git_update.main(), 0)
        self.assertEqual(order, ['check', 'update', 'check', 'update'])


class InstallationSnapshotTest(unittest.TestCase):
    def test_real_host_plan_ignores_business_files_and_detects_installed_copy_drift(self):
        import sys
        from src.automation.scheduling import installer
        from src.automation.scheduling.config import load_config
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            units, controls = directory / 'units', directory / 'controls'
            units.mkdir()
            controls.mkdir()
            config_dir = check.ROOT / 'configs/automation'
            location = dict(host='mengkjin-server', config_dir=str(config_dir))
            with (patch.object(check, 'runtime_dir', return_value=directory),
                  patch.object(installer, 'runtime_dir', return_value=directory),
                  patch.object(check, 'context', return_value=location),
                  patch.object(installer, 'UNIT_DIR', units),
                  patch.object(installer, 'desktop_path', return_value=directory / 'desktop'),
                  patch.object(installer, 'command', return_value='UTC'),
                  patch.object(installer, 'read_cron', return_value='\n'),
                  patch.object(installer, 'timer_state', return_value=dict(ActiveState='active', UnitFileState='enabled')),
                  patch('src.automation.scheduling.autofix_store.state_dir', return_value=controls),
                  patch.object(check, 'installed_config', side_effect=lambda: json.loads((directory / 'installed.json').read_text()))):
                plan = installer.build_plan(load_config(config_dir, location['host']), '', check.ROOT, Path(sys.executable), 'UTC')
                for name, content in plan['units'].items():
                    (units / name).write_text(content)
                if plan['desktop_autostart']:
                    (directory / 'desktop').write_text(plan['desktop_autostart'])
                (directory / 'installed.json').write_text(json.dumps(plan['config']))
                (directory / 'manifest.json').write_text(json.dumps(dict(units=plan['units'], cron=plan['cron'], sources=check.sources())))
                for source, target in [('worker_bootstrap.py', 'worker_bootstrap.py'), ('autofix_sandbox.py', 'policy.py')]:
                    (controls / target).write_bytes(Path(installer.__file__).with_name(source).read_bytes())
                self.assertEqual(check.snapshot()[-1], {})
                (directory / 'business.py').write_text('business change')
                self.assertEqual(check.snapshot()[-1], {})
                (controls / 'worker_bootstrap.py').write_text('stale copy')
                self.assertIn(str(controls / 'worker_bootstrap.py'), check.snapshot()[-1])


class RetimingPrivilegeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.enterContext(patch.object(privileged, 'UNITS', self.directory))
        self.service = self.directory / NAME.replace('.timer', '.service')
        self.service.write_text('[Service]\nUser=ordinary\nExecStart=/fixed/command\n')
        self.timer = self.directory / NAME
        self.timer.write_text(TIMER)
        self.policy = dict(timezone='UTC', timers={NAME: dict(envelope=privileged.envelope(TIMER),
                            service_digest=hashlib.sha256(self.service.read_bytes()).hexdigest())})

    def request(self, content):
        return dict(argv=['sudo', 'install', '-m', '0644', '/unused/staging', str(self.timer)], content=content)

    def test_only_calendar_bytes_can_change(self):
        updated = TIMER.replace('12:00', '13:00')
        privileged.execute(self.request(updated), self.policy)
        self.assertEqual(self.timer.read_text(), updated)
        privileged.execute(self.request(TIMER), self.policy)  # Rollback is equally constrained.
        for unsafe in (updated.replace('Persistent=false', 'Persistent=true'),
                       updated + '[Service]\nExecStart=/evil\n', updated.replace(' UTC', ' Asia/Shanghai'),
                       updated.replace('13:00', '29:00')):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                privileged.execute(self.request(unsafe), self.policy)
        self.assertEqual(self.timer.read_text(), TIMER)

    def test_service_changes_symlinks_and_unknown_operations_are_rejected(self):
        self.service.write_text('changed')
        with self.assertRaises(ValueError):
            privileged.execute(self.request(TIMER), self.policy)
        self.service.write_text('[Service]\nUser=ordinary\nExecStart=/fixed/command\n')
        self.timer.unlink()
        self.timer.symlink_to(self.service)
        with self.assertRaises(ValueError):
            privileged.execute(self.request(TIMER), self.policy)
        with self.assertRaises(ValueError):
            privileged.execute(dict(argv=['sudo', 'systemctl', 'start', 'unrelated.service']), self.policy)
