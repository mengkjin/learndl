"""OOM recovery uses synthetic evidence; no real OOM, mail or systemd changes."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import yaml

from src.api.calls.worklist_state import WorklistState
from src.automation.scheduling import idle_worklist as idle, recovery, runtime
from src.automation.scheduling.config import load_config
from src.automation.scheduling.installer import build_plan, validate_unit
from src.proj import PATH
from src.res.model.util import training_history
from src.res.model.util.config.config import ScheduleConfig


def require_candidate(store):
    request = idle.candidate(store)
    if not isinstance(request, dict):
        raise AssertionError('expected an idle candidate')
    return request


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(runtime, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(idle, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(idle, 'idle_reason', return_value=None))
        self.enterContext(patch.object(idle, 'settings', return_value={'enabled': True}))
        self.enterContext(patch.object(recovery, 'boot_id', return_value='boot'))
        self.enterContext(patch.object(recovery.psutil, 'virtual_memory', return_value=SimpleNamespace(total=100, available=50)))
        self.enterContext(patch.object(training_history, 'git', return_value='commit'))
        self.worklist = self.root / 'worklist.yaml'
        self.worklist.write_text('fit: [first, second]\nresume: false\nforce: true\n')
        self.enterContext(patch.object(PATH, 'sched_worklist', self.worklist))
        self.schedule = self.root / 'schedule.yaml'
        self.schedule.write_text('model.module: gru\n')
        self.enterContext(patch.object(ScheduleConfig, 'find_path', return_value=self.schedule))
        self.folder = self.root / 'model'
        self.folder.mkdir()
        self.store = runtime.RunStore()
        revisions = WorklistState.revisions('first')
        self.parent_run: dict[str, Any] = dict(id='parent', owner=idle.OWNER, task='first', phase='error',
                        revisions=revisions, version=idle.version(revisions), returncode=-9,
                        boot_id='boot', started_at=1, finished_at=100, pid=os.getpid(),
                        invocation_id='invocation', timeout_seconds=None, script_email_enabled=True)
        self.store.put(self.parent_run)
        recovery.record(self.store, self.parent_run['id'], service_result='oom-kill', model_path=str(self.folder))

    def test_exit_classification_does_not_infer_oom_from_signal(self):
        for code, reason in [(0, 'success'), (1, 'script_error'), (-11, 'native_crash'),
                             (-15, 'interrupted'), (-2, 'interrupted'), (-9, 'unknown_exit'), (None, 'unknown_exit')]:
            with self.subTest(code=code):
                recovery.record(self.store, 'parent', service_result='signal')
                self.assertEqual(recovery.exit_reason(self.store, dict(self.parent_run, returncode=code)), reason)
        recovery.record(self.store, 'parent', service_result='oom-kill')
        self.assertEqual(recovery.exit_reason(self.store, self.parent_run), 'oom')
        self.assertEqual(recovery.exit_reason(self.store, dict(self.parent_run, term_at=20)), 'watchdog_timeout')
        self.assertEqual(recovery.exit_reason(self.store, dict(self.parent_run, boot_id='previous')), 'reboot')
        for outcome in ('error', 'success', 'interrupted'):
            recovery.record(self.store, 'parent', script_outcome=outcome)
            self.assertEqual(recovery.exit_reason(self.store, self.parent_run), 'script_error' if outcome == 'error' else outcome)

    def test_stop_post_matches_exact_invocation_and_survives_store_reopen(self):
        recovery.record(self.store, 'parent', service_result='signal')
        with patch.dict(os.environ, INVOCATION_ID='old', SERVICE_RESULT='oom-kill'):
            recovery.stop_post(self.store)
        self.assertEqual(recovery.evidence(self.store, 'parent')['service_result'], 'signal')
        with patch.dict(os.environ, INVOCATION_ID='invocation', SERVICE_RESULT='oom-kill'):
            recovery.stop_post(self.store)
        restarted = runtime.RunStore(self.store.path)
        self.assertEqual(recovery.exit_reason(restarted, self.parent_run), 'oom')
        self.assertEqual(recovery.evidence(restarted, 'parent')['service_result'], 'oom-kill')
        self.assertEqual(recovery.exit_reason(restarted, dict(self.parent_run, returncode=-15)), 'interrupted')
        self.assertEqual(recovery.exit_reason(restarted, dict(self.parent_run, returncode=-15, term_at=20)), 'watchdog_timeout')

    def test_policy_cooldown_limit_disabled_and_memory_gate(self):
        self.assertEqual(recovery.recovery_plan(self.store, self.parent_run, {}, now=999)['recovery_status'], 'cooldown')
        self.assertTrue(recovery.recovery_plan(self.store, self.parent_run, {}, now=1000)['retry'])
        self.assertFalse(recovery.recovery_plan(self.store, dict(self.parent_run, retry_index=3), {}, now=1000)['retry'])
        self.assertFalse(recovery.recovery_plan(self.store, self.parent_run, {'recovery_enabled': False}, now=1000)['retry'])
        with patch.object(recovery.psutil, 'virtual_memory', return_value=SimpleNamespace(total=100, available=19)):
            self.assertEqual(recovery.recovery_plan(self.store, self.parent_run, {}, now=1000)['recovery_status'], 'system memory busy')
        self.folder.rmdir()
        self.assertIn('directory missing', recovery.recovery_plan(self.store, self.parent_run, {}, now=1000)['recovery_status'])

    def test_abnormal_exit_recovers_without_oom_evidence(self):
        recovery.record(self.store, 'parent', service_result='signal')
        for code, boot, expected in [(-9, 'boot', 'unknown_exit'), (None, 'boot', 'unknown_exit'),
                                     (-11, 'boot', 'native_crash'), (None, 'old', 'reboot')]:
            with self.subTest(code=code, boot=boot):
                run = dict(self.parent_run, returncode=code, boot_id=boot)
                self.store.put(run)
                plan = recovery.recovery_plan(self.store, run, {}, now=1000)
                self.assertEqual(plan['exit_reason'], expected)
                self.assertTrue(plan['retry'])
                request = dict(require_candidate(self.store), pid=os.getpid())
                self.assertEqual(recovery.resume_path(self.store, request, 'first', run['version']), str(self.folder))

    def test_ctrl_c_sigterm_and_normal_outcomes_never_retry_even_after_reboot(self):
        for code in (-2, -15, 130, 143, 1, 0):
            for boot in ('boot', 'old'):
                with self.subTest(code=code, boot=boot):
                    run = dict(self.parent_run, returncode=code, boot_id=boot)
                    self.assertFalse(recovery.recovery_plan(self.store, run, {}, now=1000)['retry'])
        for exit_code, status in [('killed', 'TERM'), ('killed', 'INT'), ('exited', '130'), ('exited', '143')]:
            recovery.record(self.store, 'parent', service_exit_code=exit_code, service_exit_status=status)
            run = dict(self.parent_run, returncode=None, boot_id='old')
            self.assertEqual(recovery.exit_reason(self.store, run), 'interrupted')
            self.assertFalse(recovery.recovery_plan(self.store, run, {}, now=1000)['retry'])

    def test_stop_observer_cancels_queued_retry_before_launch(self):
        recovery.record(self.store, 'parent', service_result='signal')
        idle.dispatch({'enabled': True}, self.store)
        with patch.dict(os.environ, INVOCATION_ID='invocation', SERVICE_RESULT='signal',
                        EXIT_CODE='killed', EXIT_STATUS='TERM'):
            recovery.stop_post(self.store)
        with patch.object(idle, '_launch') as launch:
            idle.worker()
            launch.assert_not_called()

    def test_cli_installer_apply_uses_unified_installation(self):
        from src.api.calls.app import ManageTaskSchedules
        with patch('src.api.calls.app.subprocess.run', return_value=SimpleNamespace(returncode=0)) as command:
            ManageTaskSchedules(action='apply').run()
        command.assert_called_once_with(['bash', str(PATH.runs / 'install_schedules.sh'), 'apply'],
                                        cwd=PATH.main, check=False)
        script = (PATH.runs / 'install_schedules.sh').read_text()
        self.assertIn('-m src.automation.scheduling.installer', script)
        config = load_config(PATH.main / 'configs/automation', 'mengkjin-server')
        plan = build_plan(config, '', PATH.main, Path(sys.executable), config['timezone'])
        self.assertTrue(plan['config']['watchdog']['jobs']['idle_worklist']['recovery_enabled'])
        self.assertIn('ExecStopPost=', plan['units']['learndl-idle-worklist.service'])

    def test_latest_attempt_recovery_then_error_blocks_older_oom(self):
        request = require_candidate(self.store)
        self.assertEqual(request['parent_run_id'], 'parent')
        self.assertEqual(request['resume_model_path'], str(self.folder))
        self.assertEqual(request['retry_index'], 1)
        failed = dict(request, id='retry', owner=idle.OWNER, phase='error', returncode=1, finished_at=100)
        self.store.put(failed)
        self.assertEqual(require_candidate(self.store)['task'], 'second')
        failed.update(phase='complete', returncode=0)
        self.store.put(failed)
        WorklistState('first').save(status='success', model_path=str(self.folder), revisions=request['revisions'])
        self.assertEqual(require_candidate(self.store)['task'], 'second')

    def test_cooling_schedule_does_not_block_other_work(self):
        self.parent_run['finished_at'] = recovery.time.time()
        self.store.put(self.parent_run)
        self.assertEqual(require_candidate(self.store)['task'], 'second')

    def test_three_retries_then_stop_and_new_version_is_fresh(self):
        for number in range(1, 4):
            request = require_candidate(self.store)
            self.assertEqual(request['retry_index'], number)
            request.update(id=f'retry-{number}', owner=idle.OWNER, phase='error', returncode=-9,
                           finished_at=100, boot_id='boot')
            self.store.put(request)
            recovery.record(self.store, request['id'], service_result='oom-kill', model_path=str(self.folder))
        self.assertEqual(require_candidate(self.store)['task'], 'second')
        self.worklist.write_text(self.worklist.read_text() + '# new version\n')
        fresh = require_candidate(self.store)
        self.assertEqual(fresh['task'], 'first')
        self.assertNotIn('parent_run_id', fresh)

    def test_worker_rechecks_recovery_and_does_not_duplicate_queue(self):
        idle.dispatch({'enabled': True}, self.store)
        self.assertEqual(idle.dispatch({'enabled': True}, self.store)['idle_worklist'], 'queued or active')
        with patch.object(idle, '_launch', return_value=0) as launch:
            recovery.record(self.store, 'parent', script_outcome='error')
            idle.worker()
            launch.assert_not_called()

    def test_worker_rechecks_memory_and_configuration_after_queueing(self):
        idle.dispatch({'enabled': True}, self.store)
        with patch.object(recovery.psutil, 'virtual_memory', return_value=SimpleNamespace(total=100, available=10)), \
             patch.object(idle, '_launch') as launch:
            idle.worker()
            launch.assert_not_called()
        idle.dispatch({'enabled': True}, self.store)
        self.worklist.write_text(self.worklist.read_text() + '# changed\n')
        with patch.object(idle, '_launch') as launch:
            idle.worker()
            launch.assert_not_called()

    def test_removed_resume_directory_emits_one_blocked_notification(self):
        self.assertEqual(require_candidate(self.store)['task'], 'first')
        self.folder.rmdir()
        self.assertEqual(require_candidate(self.store)['task'], 'second')
        idle.candidate(self.store)
        with self.store.connect() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM events WHERE id LIKE '%:recovery_blocked'").fetchone()[0], 1)

    def test_outcome_persisted_before_smtp_exception(self):
        from datetime import datetime
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.util.web.emailer import Email
        task = SimpleNamespace(email=True, kwargs={}, execution_status='Error', task_name='schedule',
                               time_str='now', source='py', task_full_name='schedule',
                               init_time=datetime.now(), end_time=datetime.now(), status='Error',
                               error_message='setup error', exit_message='', exit_files=[])
        with patch.dict(os.environ, LEARNDL_MANAGED_RUN='parent'), \
             patch.object(Email, 'send', side_effect=OSError('SMTP unavailable')):
            with self.assertRaises(OSError):
                AutoRunTask.send_email(cast(AutoRunTask, task))
        self.assertEqual(recovery.exit_reason(self.store, self.parent_run), 'script_error')
        self.assertFalse(recovery.recovery_plan(self.store, self.parent_run, {})['retry'])

    def test_recovery_options_reject_invalid_types_and_bounds(self):
        directory = self.root / 'config'
        shutil.copytree(PATH.main / 'configs/automation', directory)
        path = directory / 'maintenance.yaml'
        original = yaml.safe_load(path.read_text())
        for key, value in [('recovery_enabled', 'yes'), ('recovery_max_retries', -1),
                           ('recovery_max_retries', True), ('recovery_delay_seconds', 0),
                           ('recovery_min_available_memory_percent', 101)]:
            with self.subTest(key=key, value=value):
                config = yaml.safe_load(yaml.safe_dump(original))
                config['watchdog']['jobs']['idle_worklist'][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.assertRaises(ValueError):
                    load_config(directory, 'mengkjin-server')

    def test_resume_path_rejects_replaced_identity_and_missing_directory(self):
        request = dict(require_candidate(self.store), pid=os.getpid())
        self.assertEqual(recovery.resume_path(self.store, request, 'first', self.parent_run['version']), str(self.folder))
        with self.assertRaises(RuntimeError):
            recovery.resume_path(self.store, dict(request, pid=-1), 'first', self.parent_run['version'])
        self.folder.rmdir()
        with self.assertRaises(FileNotFoundError):
            recovery.resume_path(self.store, request, 'first', self.parent_run['version'])

    def test_progress_outcome_is_durable_without_email(self):
        recovery.child_progress('parent', dict(run_id='training', status='failed', model_path=str(self.folder)), self.store)
        # A stale launcher snapshot must not erase the independently persisted error.
        self.store.put(self.parent_run)
        self.assertEqual(recovery.exit_reason(self.store, self.parent_run), 'script_error')

    def test_error_mail_includes_recovery_reason_and_deduplicates(self):
        runtime.finish_idle_event(self.store, self.parent_run, 'abnormal exit')
        sender = Mock(return_value=True)
        with patch('src.api.util.backend.task.TaskDatabase'):
            runtime.deliver_timeout_events(sender, self.store)
            runtime.deliver_timeout_events(sender, self.store)
        sender.assert_called_once()
        self.assertIn('"exit_reason": "oom"', sender.call_args.args[1])
        self.assertIn('"retry_index": 1', sender.call_args.args[1])

    def test_installer_stop_observer_is_fixed_and_old_units_remain_valid(self):
        config = load_config(PATH.main / 'configs/automation', 'mengkjin-server')
        plan = build_plan(config, '', PATH.main, Path(sys.executable), local_timezone=config['timezone'])
        name = 'learndl-idle-worklist.service'
        unit = plan['units'][name]
        self.assertIn('ExecStopPost=', unit)
        validate_unit(name, unit)
        validate_unit(name, '\n'.join(line for line in unit.splitlines() if not line.startswith('ExecStopPost=')))
        with self.assertRaises(ValueError):
            validate_unit(name, unit.replace('src.automation.scheduling.recovery', 'evil.module'))
        with self.assertRaises(ValueError):
            validate_unit(name, unit + 'ExecStopPost=/bin/true\n')
