from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from src.api.task_monitor.scheduling.config import duration, load_config, schedule_slots
from src.api.task_monitor.scheduling.installer import apply_plan, build_plan, cron_coverage, external_cron, service, validate_unit
from src.api.task_monitor.scheduling.runtime import RunStore, inspect_timeouts, members
from src.api.task_monitor.watchdog import default_jobs


ROOT = Path(__file__).resolve().parents[4]


class ScheduleConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'runs/scheduling', 'mengkjin-server')

    def test_timeouts_and_registry(self):
        self.assertEqual(self.config['tasks']['daily_update']['timeout_seconds'], 43200)
        self.assertEqual(self.config['tasks']['weekly_update']['timeout_seconds'], 259200)
        self.assertIsNone(duration(None))
        for value in ('0h', '-1s', 'abc', 100):
            with self.assertRaises(ValueError):
                duration(value)
        self.assertEqual(default_jobs({'jobs': {}}), ())

    def test_partial_cron_comment_and_restore(self):
        line = f'30 18 * * * /bin/bash {ROOT}/runs/daily_update.sh\n'
        plan = build_plan(self.config, line, ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')
        self.assertEqual(plan['report']['daily_update']['cron_preserved'], 7)
        self.assertEqual(plan['report']['daily_update']['managed_slots'], 28)
        self.assertNotIn('18:30:00', plan['units']['learndl-schedule-daily_update.timer'])
        commented = build_plan(self.config, '# ' + line, ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')
        self.assertEqual(commented['report']['daily_update']['managed_slots'], 35)
        self.assertIn('18:30:00', commented['units']['learndl-schedule-daily_update.timer'])

    def test_force_arguments_and_multiple_times(self):
        lines = f'30 18,20 * * * /bin/bash {ROOT}/runs/daily_update.sh\n0 8 * * * /bin/bash {ROOT}/runs/daily_update.sh --forfeit_if_done=False\n'
        coverage, conflicts, _ = cron_coverage(lines, self.config, ROOT, 'Asia/Shanghai')
        self.assertFalse(conflicts)
        self.assertEqual(len(coverage['daily_update']), 14)
        self.assertEqual(len(coverage['daily_update_force']), 7)

    def test_external_cron_preserved_and_ambiguous_project_blocked(self):
        line = '0 20 * * * /home/user/bin/check_onedrive.sh\n'
        plan = build_plan(self.config, line, ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')
        self.assertTrue(plan['cron'].startswith(line))
        self.assertFalse(plan['conflicts'])
        bad = f'0 8 * * * /bin/bash {ROOT}/runs/daily_update.sh && touch /tmp/test\n'
        self.assertTrue(build_plan(self.config, bad, ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')['conflicts'])

    def test_duplicate_and_timezone_conflicts(self):
        line = f'30 18 * * * /bin/bash {ROOT}/runs/daily_update.sh\n'
        self.assertTrue(cron_coverage(line * 2, self.config, ROOT, 'Asia/Shanghai')[1])
        self.assertTrue(cron_coverage(line, self.config, ROOT, 'UTC')[1])

    def test_managed_cron_does_not_cover_itself(self):
        config = copy.deepcopy(self.config)
        config['default_backend'] = 'cron'
        for task in config['tasks'].values():
            task['backend'] = 'cron'
        plan = build_plan(config, '# external\n', ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')
        repeated = build_plan(config, plan['cron'], ROOT, Path('/venv/bin/python'), 'Asia/Shanghai')
        self.assertEqual(plan['cron'], repeated['cron'])
        self.assertEqual(external_cron(plan['cron']), '# external\n')

    def test_invalid_calendar(self):
        for schedule in ({'times': ['25:00']}, {'every_minutes': 0}, {'every_minutes': 7}):
            with self.assertRaises(ValueError):
                schedule_slots(schedule)

    def test_manifest_cannot_install_privileged_or_injected_service(self):
        valid = service(ROOT, Path(sys.executable), 'src.api.task_monitor.scheduling.runner', ['run', 'daily_update'])
        validate_unit('learndl-schedule-daily_update.service', valid)
        for changed in (valid + 'ExecStartPre=/bin/evil\n',
                        valid.replace('After=network-online.target', 'After=poweroff.target'),
                        valid.replace('NoNewPrivileges=true', 'NoNewPrivileges=false'),
                        valid + 'Group=root\n'):
            with self.assertRaises(ValueError):
                validate_unit('learndl-schedule-daily_update.service', changed)


class TimeoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RunStore(Path(self.temp.name) / 'runs.sqlite')
        self.run_record = {'id': 'run1', 'task': 'daily_update', 'phase': 'running', 'pid': 100,
                    'uid': 123, 'process_created': 1., 'started_at': 100., 'timeout_seconds': 10}
        self.store.put(self.run_record)

    def test_term_grace_then_kill_without_sleeping(self):
        target = 'src.api.task_monitor.scheduling.runtime.'
        with patch(target + 'members', return_value=[MagicMock()]), patch(target + 'signal_members') as send:
            inspect_timeouts(self.store, now=109)
            send.assert_not_called()
            inspect_timeouts(self.store, now=110)
            self.assertEqual(send.call_args.args[2], signal.SIGTERM)
            inspect_timeouts(self.store, now=139)
            self.assertEqual(send.call_args.args[2], signal.SIGTERM)
            inspect_timeouts(self.store, now=140)
            self.assertEqual(send.call_args.args[2], signal.SIGKILL)
        self.assertEqual(self.store.get('run1')['phase'], 'terminating')
        with patch(target + 'members', return_value=[]), patch('src.api.util.backend.task.TaskDatabase'):
            inspect_timeouts(self.store, now=141)
            inspect_timeouts(self.store, now=200)
        self.assertEqual(self.store.get('run1')['phase'], 'killed')
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM events').fetchone()[0], 1)

    def test_permission_failure_is_not_marked_killed(self):
        with patch('src.api.task_monitor.scheduling.runtime.members', side_effect=psutil.AccessDenied(100)):
            inspect_timeouts(self.store, now=110)
            inspect_timeouts(self.store, now=170)
        self.assertEqual(self.store.get('run1')['phase'], 'running')
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM events').fetchone()[0], 1)

    def test_pid_reuse_never_signalled(self):
        with patch('src.api.task_monitor.scheduling.runtime.psutil.Process') as process:
            process.return_value.create_time.return_value = 2.
            with self.assertRaisesRegex(RuntimeError, 'reused'):
                members(self.run_record)

    def test_failed_database_finalization_is_retried(self):
        run: dict = dict(self.run_record, phase='terminating', term_at=110)
        self.store.put(run)
        self.store.link('task1', run['id'])
        task = MagicMock(pid=run['pid'])
        with patch('src.api.task_monitor.scheduling.runtime.members', return_value=[]), \
             patch('src.api.util.backend.task.TaskDatabase') as database:
            database.return_value.get_task.return_value = task
            database.return_value.update_task.side_effect = RuntimeError('temporary DB failure')
            inspect_timeouts(self.store, now=150)
            self.assertEqual(self.store.get(run['id'])['phase'], 'finalizing')
            database.return_value.update_task.side_effect = None
            inspect_timeouts(self.store, now=210)
        self.assertEqual(self.store.get(run['id'])['phase'], 'killed')
        self.assertEqual(self.store.get(run['id'])['finished_at'], 150)

    def test_normal_finish_and_no_timeout(self):
        with patch('src.api.task_monitor.scheduling.runtime.members', return_value=[]):
            inspect_timeouts(self.store, now=105)
        self.assertEqual(self.store.get('run1')['phase'], 'error')
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM events').fetchone()[0], 0)

    def test_live_process_timeout_and_retry_email(self):
        # A real isolated process ignores TERM, exercising escalation without
        # running project business code or sending mail.
        proc = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'],
                                start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout is not None
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            run = dict(self.run_record, pid=proc.pid, process_created=psutil.Process(proc.pid).create_time(), uid=os.getuid())
            self.store.put(run)
            inspect_timeouts(self.store, now=110)
            self.assertIsNone(proc.poll())
            inspect_timeouts(self.store, now=140)
            proc.wait(timeout=5)
            with patch('src.api.util.backend.task.TaskDatabase'):
                inspect_timeouts(self.store, now=141)
                from src.api.task_monitor.scheduling.runtime import deliver_timeout_events
                send = MagicMock(side_effect=[False, True])
                self.assertFalse(deliver_timeout_events(send, self.store))
                self.assertTrue(deliver_timeout_events(send, self.store))
                self.assertTrue(deliver_timeout_events(send, self.store))
                self.assertEqual(send.call_count, 2)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            if proc.stdout:
                proc.stdout.close()


class InstallerTransactionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.units = self.root / 'units'
        self.units.mkdir()
        self.state = self.root / 'state'
        self.state.mkdir()
        self.cron = '# unrelated\n0 20 * * * /home/user/bin/check_onedrive.sh\n'
        self.calls = []
        for name, value in [('UNIT_DIR', self.units), ('read_cron', lambda: self.cron), ('command', self.execute)]:
            patcher = patch('src.api.task_monitor.scheduling.installer.' + name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('src.api.task_monitor.scheduling.installer.psutil.process_iter', return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('src.api.task_monitor.scheduling.installer.timer_state', return_value={'ActiveState': 'active', 'UnitFileState': 'enabled'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = load_config(ROOT / 'runs/scheduling', 'mengkjin-server')

    def execute(self, argv, *, input_text=None):
        self.calls.append(argv)
        if argv[:2] == ['sudo', 'install']:
            shutil.copyfile(argv[-2], argv[-1])
        elif argv[:2] == ['sudo', 'mv']:
            Path(argv[-2]).rename(argv[-1])
        elif argv == ['crontab', '-']:
            assert isinstance(input_text, str)
            self.cron = input_text
        return ''

    def plan(self):
        return build_plan(copy.deepcopy(self.config), self.cron, ROOT, Path(sys.executable), 'Asia/Shanghai')

    def test_apply_idempotence_and_first_install_rollback(self):
        original = self.cron
        apply_plan(self.plan(), self.cron, self.state)
        count = len(self.calls)
        apply_plan(self.plan(), self.cron, self.state)
        self.assertEqual(len(self.calls), count)
        backup = json.loads((self.state / 'rollback.json').read_text())
        rollback = {'config': backup['config'], 'units': backup['manifest']['units'],
                    'cron': backup['cron'], 'conflicts': {}, 'global_conflicts': [], 'report': {}}
        apply_plan(rollback, self.cron, self.state)
        self.assertEqual(self.cron, original)
        self.assertFalse((self.state / 'installed.json').exists())
        self.assertFalse(list(self.units.glob('*.timer')))
        self.assertTrue(list(self.units.glob('*.learndl-disabled')))

    def test_partial_install_resume_preserves_original_backup(self):
        original = self.cron
        execute = self.execute
        failed = False

        def fail_once(argv, **kwargs):
            nonlocal failed
            if argv[:2] == ['sudo', 'install'] and not failed:
                failed = True
                raise RuntimeError('simulated installation interruption')
            return execute(argv, **kwargs)

        with patch('src.api.task_monitor.scheduling.installer.command', side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                apply_plan(self.plan(), self.cron, self.state)
        self.assertEqual(json.loads((self.state / 'transaction.json').read_text())['phase'], 'paused')
        apply_plan(self.plan(), self.cron, self.state)
        self.assertEqual(json.loads((self.state / 'transaction.json').read_text())['phase'], 'complete')
        self.assertEqual(json.loads((self.state / 'rollback.json').read_text())['cron'], original)

    def test_changed_cron_aborts_before_mutation(self):
        plan = self.plan()
        original = self.cron
        self.cron += '# concurrently edited\n'
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            apply_plan(plan, original, self.state)
        self.assertFalse(list(self.units.iterdir()))


class RunnerTest(unittest.TestCase):
    def test_real_runner_registers_and_releases_lock(self):
        from src.api.task_monitor.scheduling.runner import run
        from src.proj.util.script.script_lock import ScriptLockMultiple
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = load_config(ROOT / 'runs/scheduling', 'mengkjin-server')
            fixture = str(ROOT / 'src/api/task_monitor/tests/fixtures/scheduler_probe.py')
            with patch('src.api.task_monitor.scheduling.runner.installed_config', return_value=config), \
                 patch('src.api.task_monitor.scheduling.runtime.runtime_dir', return_value=root), \
                 patch.object(ScriptLockMultiple, 'LOCK_DIR', root / 'locks'), \
                 patch.dict('src.api.task_monitor.scheduling.runner.ENTRYPOINTS', {'daily_update': fixture}):
                self.assertEqual(run('daily_update'), 0)
                store = RunStore(root / 'runs.sqlite')
                records = store.active()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['returncode'], 0)
                self.assertEqual(records[0]['timeout_seconds'], 43200)
                inspect_timeouts(store)
                self.assertFalse(store.active())
                with ScriptLockMultiple('daily_update', 1, timeout=1, wait_time=2):
                    pass

    def test_locked_skip_does_not_start_or_register_process(self):
        from src.api.task_monitor.scheduling.runner import run
        from src.proj.util.script.script_lock import ScriptLockMultiple
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = load_config(ROOT / 'runs/scheduling', 'mengkjin-server')
            with patch('src.api.task_monitor.scheduling.runner.installed_config', return_value=config), \
                 patch.object(ScriptLockMultiple, 'LOCK_DIR', root / 'locks'), \
                 patch('src.api.task_monitor.scheduling.runner.subprocess.Popen') as start:
                with ScriptLockMultiple('daily_update', 1, timeout=1):
                    self.assertEqual(run('daily_update'), 0)
                start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
