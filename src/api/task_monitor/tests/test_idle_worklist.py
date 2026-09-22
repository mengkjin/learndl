"""Idle dispatch, durable supervision and shared fit occupancy without GPUs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import portalocker

from src.api.calls.worklist_state import WorklistState
from src.api.task_monitor.scheduling import idle_worklist as idle
from src.api.task_monitor.scheduling import runtime
from src.api.task_monitor.scheduling.config import load_config
from src.api.task_monitor.scheduling.installer import build_plan, validate_unit
from src.proj import PATH
from src.proj.util.script import FitLock, FitLockNN
from src.res.model.util import training_history
from src.res.model.util.config.config import ScheduleConfig


class IdleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(idle, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(runtime, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(training_history, 'git', return_value='test-commit'))
        self.worklist = self.root / 'worklist.yaml'
        self.worklist.write_text('fit: [first, second]\nresume: true\nforce: true\n')
        self.enterContext(patch.object(PATH, 'sched_worklist', self.worklist))
        self.schedule = self.root / 'schedule.yaml'
        self.schedule.write_text('model.module: gru\n')
        self.enterContext(patch.object(ScheduleConfig, 'find_path', return_value=self.schedule))
        self.store = runtime.RunStore()
        self.enterContext(patch.object(idle, 'idle_reason', return_value=None))
        self.folder = self.root / 'model'
        self.folder.mkdir()
        self.options = {'enabled': True, 'progress_timeout_seconds': 43200}

    def save_success(self, name='first'):
        WorklistState(name).save(status='success', revisions=WorklistState.revisions(name), model_path=str(self.folder))

    def test_missing_schedules_after_valid_candidate_are_all_reported(self):
        self.worklist.write_text('fit: [first, mincr, missing, mincr]\nforce: true\n')
        with patch.object(ScheduleConfig, 'find_path', side_effect=lambda name: self.schedule if name == 'first' else None):
            result = idle.dispatch(self.options, self.store)
        self.assertEqual(result['queued'], 'first')
        self.assertEqual(result['missing_schedules'], ['mincr', 'missing'])
        self.assertEqual(len(self.store.owned(idle.OWNER)), 1)
        sender = Mock(return_value=True)
        with patch('src.api.util.backend.task.TaskDatabase') as database:
            self.assertTrue(runtime.deliver_timeout_events(sender, self.store))
            database.assert_not_called()
        sender.assert_called_once()
        subject, body = sender.call_args.args
        self.assertIn('configuration error', subject)
        for detail in ('mincr, missing', str(self.worklist), 'test-commit', str(PATH.sched), str(PATH.sched_shared)):
            self.assertIn(detail, body)

    def test_missing_schedule_reports_while_gpu_busy_or_another_run_active(self):
        for active in (False, True):
            with self.subTest(active=active):
                if active:
                    self.store.put(dict(id='existing', owner=idle.OWNER, task='old', phase='running'))
                with patch.object(ScheduleConfig, 'find_path', return_value=None), \
                     patch.object(idle, 'idle_reason', return_value='GPU memory busy: 30.0%'):
                    result = idle.dispatch(self.options, self.store)
                self.assertEqual(result['missing_schedules'], ['first', 'second'])
                self.assertEqual(result['idle_worklist'], 'queued or active' if active else 'GPU memory busy: 30.0%')
        sender = Mock(return_value=True)
        self.assertTrue(runtime.deliver_timeout_events(sender, self.store))
        sender.assert_called_once()

    def test_missing_schedule_mail_retries_and_deduplicates_across_restart(self):
        with patch.object(ScheduleConfig, 'find_path', return_value=None):
            idle.dispatch(self.options, self.store)
            sender = Mock(side_effect=[False, OSError('SMTP unavailable'), True])
            self.assertFalse(runtime.deliver_timeout_events(sender, self.store))
            self.assertFalse(runtime.deliver_timeout_events(sender, self.store))
            restarted = runtime.RunStore(self.store.path)
            idle.dispatch(self.options, restarted)
            self.assertTrue(runtime.deliver_timeout_events(sender, restarted))
            idle.dispatch(self.options, restarted)
            self.assertTrue(runtime.deliver_timeout_events(sender, restarted))
            self.assertEqual(sender.call_count, 3)
            self.assertEqual(restarted.owned(idle.OWNER), [])
            # A new worklist revision can notify again if it is still invalid.
            self.worklist.write_text(self.worklist.read_text() + '# new revision\n')
            idle.dispatch(self.options, restarted)
            sender.side_effect = None
            sender.return_value = True
            self.assertTrue(runtime.deliver_timeout_events(sender, restarted))
            self.assertEqual(sender.call_count, 4)

    def test_adding_missing_config_allows_training_without_failure_suppression(self):
        self.worklist.write_text('fit: [first]\n')
        with patch.object(ScheduleConfig, 'find_path', return_value=None):
            result = idle.dispatch(self.options, self.store)
        self.assertEqual(result['missing_schedules'], ['first'])
        sender = Mock(return_value=True)
        runtime.deliver_timeout_events(sender, self.store)
        result = idle.dispatch(self.options, self.store)
        self.assertEqual(result['queued'], 'first')
        self.assertNotIn('missing_schedules', result)
        runtime.deliver_timeout_events(sender, self.store)
        sender.assert_called_once()

    def test_one_queued_task_no_duplicate_and_force_once(self):
        self.assertEqual(idle.dispatch(self.options, self.store)['queued'], 'first')
        self.assertIn('idle_worklist', idle.dispatch(self.options, self.store))
        self.assertEqual(len(self.store.owned(idle.OWNER)), 1)
        request = self.store.owned(idle.OWNER)[0]
        request.update(phase='complete')
        self.store.put(request)
        self.save_success()
        self.assertEqual(idle.dispatch(self.options, self.store)['queued'], 'second')

    def test_failure_suppression_reset_and_configuration_change(self):
        idle.dispatch(self.options, self.store)
        request = self.store.owned(idle.OWNER)[0]
        with self.assertRaisesRegex(RuntimeError, 'active'):
            idle.reset('first', self.store)
        request['phase'] = 'error'
        self.store.put(request)
        self.assertEqual(idle.candidate(self.store)['task'], 'second')
        idle.reset('first', self.store)
        self.assertEqual(idle.candidate(self.store)['task'], 'first')
        request['retry_allowed'] = False
        self.store.put(request)
        self.schedule.write_text('model.module: gru\n# changed\n')
        self.assertEqual(idle.candidate(self.store)['task'], 'first')

    def test_missing_directory_retrains_completed_resume_true(self):
        self.save_success()
        self.assertEqual(idle.candidate(self.store)['task'], 'second')
        self.folder.rmdir()
        self.assertEqual(idle.candidate(self.store)['task'], 'first')

    def test_manual_schedule_lock_defers_only_that_schedule(self):
        with WorklistState('first').lock():
            self.assertEqual(idle.candidate(self.store)['task'], 'second')

    def test_worker_rechecks_queue_version_and_disabled_setting(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                idle.dispatch(self.options, self.store)
                options = dict(self.options, enabled=not disabled)
                if not disabled:
                    self.schedule.write_text(self.schedule.read_text() + '# edit\n')
                with patch.object(idle, 'settings', return_value=options), patch.object(idle, '_launch') as launch:
                    self.assertEqual(idle.worker(), 0)
                    launch.assert_not_called()
                self.assertTrue(all(run['phase'] == 'deferred' for run in self.store.owned(idle.OWNER)))

    def test_worker_launches_exactly_one_and_does_not_chain(self):
        idle.dispatch(self.options, self.store)
        with patch.object(idle, 'settings', return_value=self.options), patch.object(idle, '_launch', return_value=0) as launch:
            self.assertEqual(idle.worker(), 0)
            launch.assert_called_once()
            self.assertEqual(launch.call_args.args[0]['task'], 'first')
            self.assertEqual(len(self.store.owned(idle.OWNER)), 1)

    def test_launch_registration_gate_and_start_mail_retry(self):
        idle.dispatch(self.options, self.store)
        request = self.store.owned(idle.OWNER)[0]
        popen = subprocess.Popen
        def probe(command, **kwargs):
            # A real process can produce output only after its identity is durable.
            code = ('import os; fd=int(os.environ["LEARNDL_GATE_FD"]); '
                    'assert os.read(fd,1)==b"1"; os.close(fd); print("fit done", flush=True)')
            return popen([sys.executable, '-u', '-c', code], **kwargs)
        with patch.object(idle.subprocess, 'Popen', side_effect=probe):
            self.assertEqual(idle._launch(request, self.store), 0)
        runtime.inspect_timeouts(self.store)
        self.assertEqual(self.store.get(request['id'])['phase'], 'complete')
        self.assertIn('fit done', Path(request['log_path']).read_text())
        with patch('src.api.util.backend.task.TaskDatabase'):
            send = Mock(return_value=False)
            self.assertFalse(runtime.deliver_timeout_events(send, self.store))
            send.return_value = True
            self.assertTrue(runtime.deliver_timeout_events(send, self.store))
            self.assertEqual(send.call_count, 2)  # Only start; success is mailed by the script.
            runtime.inspect_timeouts(self.store)
            runtime.deliver_timeout_events(send, self.store)
            self.assertEqual(send.call_count, 2)

    def test_script_mail_receipts_and_watchdog_fallback(self):
        from datetime import datetime
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.util.web.emailer import Email
        for status, delivered, code, expected in [
            ('Success', True, 0, False), ('Success', False, 0, False),
            ('Error', True, 1, False), ('Error', False, 1, True),
            ('Success', True, 1, True), ('Error', True, -9, True),
        ]:
            with self.subTest(status=status, delivered=delivered, code=code):
                run = dict(id=f'{status}-{delivered}-{code}', owner=idle.OWNER, task='first',
                           phase='complete' if code == 0 else 'error', pid=os.getpid(), script_email_enabled=True,
                           returncode=code, started_at=10, timeout_seconds=None)
                self.store.put(run)
                task = SimpleNamespace(email=True, execution_status=status, task_name='schedule',
                                       time_str='now', source='py', task_full_name='schedule',
                                       init_time=datetime.now(), end_time=datetime.now(), status=status,
                                       error_message='', exit_message='', exit_files=[])
                with patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': run['id']}), \
                     patch.object(Email, 'send', return_value=delivered):
                    AutoRunTask.send_email(task)
                self.assertEqual(runtime.idle_exit_needs_email(self.store, run), expected)
                runtime.finish_idle_event(self.store, run, 'exit')
                send = Mock(return_value=True)
                with patch('src.api.util.backend.task.TaskDatabase'):
                    runtime.deliver_timeout_events(send, self.store)
                    runtime.finish_idle_event(self.store, run, 'exit again')
                    runtime.deliver_timeout_events(send, self.store)
                self.assertEqual(send.call_count, int(expected))

    def test_smtp_exception_leaves_error_for_watchdog_and_timeout_bypasses_receipt(self):
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.util.web.emailer import Email
        from datetime import datetime
        run = dict(id='mail-failed', owner=idle.OWNER, task='first', phase='error', pid=os.getpid(),
                   returncode=1, started_at=10, timeout_seconds=None)
        self.store.put(run)
        task = SimpleNamespace(email=True, execution_status='Error', task_name='schedule',
                               time_str='now', source='py', task_full_name='schedule',
                               init_time=datetime.now(), end_time=datetime.now(), status='Error',
                               error_message='', exit_message='', exit_files=[])
        with patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': run['id']}), \
             patch.object(Email, 'send', side_effect=OSError('SMTP unavailable')):
            with self.assertRaises(OSError):
                AutoRunTask.send_email(task)
            self.assertTrue(runtime.idle_exit_needs_email(self.store, run))
            runtime.record_script_email(success=False)
        run.update(phase='killed', term_at=20)
        self.assertTrue(runtime.idle_exit_needs_email(self.store, run))
        self.store.put(run)
        with patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': run['id']}), patch.object(Email, 'send') as send:
            AutoRunTask.send_email(task)
            send.assert_not_called()

    def test_pending_new_success_and_reported_error_mails_are_suppressed(self):
        for phase, code in [('complete', 0), ('deferred', 75), ('error', 1)]:
            run = dict(id=phase, owner=idle.OWNER, task='first', phase=phase, pid=os.getpid(), script_email_enabled=True,
                       returncode=code, started_at=10, timeout_seconds=None)
            self.store.put(run)
            self.store.event(run, 'finished', 'old pending event')
            if phase == 'error':
                with patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': run['id']}):
                    runtime.record_script_email(success=False)
        with patch('src.api.util.backend.task.TaskDatabase'):
            send = Mock(return_value=True)
            self.assertTrue(runtime.deliver_timeout_events(send, self.store))
            send.assert_not_called()
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM events WHERE sent=-1').fetchone()[0], 3)

    def test_legacy_running_task_keeps_result_mail_and_delivery_retry(self):
        run = dict(id='legacy', owner=idle.OWNER, task='first', phase='running', pid=100,
                   returncode=0, started_at=10, timeout_seconds=None)
        self.store.put(run)
        with patch.object(runtime, 'members', return_value=[]), patch('src.api.util.backend.task.TaskDatabase'):
            runtime.inspect_timeouts(self.store, now=100)
            send = Mock(side_effect=[True, False, True])  # Start succeeds, result retries.
            self.assertFalse(runtime.deliver_timeout_events(send, self.store))
            self.assertTrue(runtime.deliver_timeout_events(send, self.store))
            runtime.inspect_timeouts(self.store, now=101)
            runtime.deliver_timeout_events(send, self.store)
            self.assertEqual(send.call_count, 3)
            self.assertIn('finished', send.call_args.args[0])

    def test_legacy_result_mail_suppressed_only_when_script_already_sent(self):
        run = dict(id='legacy-receipt', owner=idle.OWNER, task='first', phase='complete',
                   pid=os.getpid(), returncode=0, started_at=10, timeout_seconds=None)
        self.store.put(run)
        self.assertTrue(runtime.idle_exit_needs_email(self.store, run))
        with patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': run['id']}):
            runtime.record_script_email(success=True)
        self.assertFalse(runtime.idle_exit_needs_email(self.store, run))

    def test_legacy_previously_suppressed_result_is_recovered_once(self):
        run = dict(id='legacy-suppressed', owner=idle.OWNER, task='first', phase='complete',
                   returncode=0, started_at=10, timeout_seconds=None)
        self.store.put(run)
        self.store.event(run, 'finished', 'legacy result')
        with self.store.connect() as connection:
            connection.execute('UPDATE events SET sent=-1')
        with patch('src.api.util.backend.task.TaskDatabase'):
            send = Mock(return_value=True)
            for _ in range(2):
                runtime.inspect_timeouts(self.store, now=100)
                runtime.deliver_timeout_events(send, self.store)
            send.assert_called_once()

    def test_no_output_timeout_then_term_and_kill(self):
        output = self.root / 'output.log'
        output.write_text('progress\n')
        os.utime(output, (1000, 1000))
        run = dict(id='run', owner=idle.OWNER, task='first', phase='running', started_at=10,
                   pid=100, uid=os.getuid(), process_created=10, timeout_seconds=None,
                   progress_timeout_seconds=43200, log_path=str(output), revisions={}, version='v')
        self.store.put(run)
        with patch.object(runtime, 'members', return_value=[Mock()]), patch.object(runtime, 'signal_members') as send:
            runtime.inspect_timeouts(self.store, now=44199)
            send.assert_not_called()
            runtime.inspect_timeouts(self.store, now=44200)
            self.assertEqual(send.call_args.args[-1], signal.SIGTERM)
            # Late output cannot cancel an already persisted termination intent.
            os.utime(output, (44201, 44201))
            runtime.inspect_timeouts(self.store, now=44230)
            self.assertEqual(send.call_args.args[-1], signal.SIGKILL)
        with patch.object(runtime, 'members', return_value=[]), patch('src.api.util.backend.task.TaskDatabase'):
            runtime.inspect_timeouts(self.store, now=44231)
        self.assertEqual(self.store.get('run')['phase'], 'killed')

    def test_reboot_ignores_reused_pid_and_reports_abnormal_exit(self):
        run = dict(id='run', owner=idle.OWNER, task='first', phase='running', started_at=10,
                   pid=os.getpid(), runner_pid=os.getpid(), boot_id='old-boot', timeout_seconds=None,
                   revisions={}, version='v')
        self.store.put(run)
        with patch.object(runtime, 'boot_id', return_value='new-boot'):
            self.assertEqual(runtime.members(run), [])
            runtime.inspect_timeouts(self.store, now=100)
        self.assertEqual(self.store.get('run')['phase'], 'error')
        with self.store.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM events').fetchone()[0], 2)


class FitOccupancyTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(FitLock, 'LOCK_DIR', self.root))
        self.enterContext(patch.object(FitLockNN, 'LOCK_DIR', self.root))
        self.enterContext(patch.object(FitLockNN, 'enabled', return_value=True))

    def test_shared_fits_are_unlimited_and_nn_stays_exclusive(self):
        self.assertFalse(FitLock.is_held())
        with FitLock.guard(), FitLock.guard():
            self.assertTrue(FitLock.is_held())
            self.assertFalse(FitLockNN.is_held())
        with FitLockNN.guard():
            self.assertTrue(FitLock.is_held())  # Old running NN compatibility.
            with FitLockNN.lock_path().open('a') as stream:
                with self.assertRaises(portalocker.LockException):
                    portalocker.lock(stream, portalocker.LOCK_EX | portalocker.LOCK_NB)
        self.assertFalse(FitLock.is_held())
        with FitLockNN.guard(try_cuda=False):
            self.assertFalse(FitLockNN.is_held())

    def test_process_death_releases_shared_occupancy(self):
        code = ('import portalocker,sys,time; f=open(sys.argv[1],"a"); '
                'portalocker.lock(f,portalocker.LOCK_SH); print("ready",flush=True); time.sleep(60)')
        proc = subprocess.Popen([sys.executable, '-c', code, str(FitLock.lock_path())], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            with FitLock.guard():
                self.assertTrue(FitLock.is_held())
            proc.kill()
            proc.wait(timeout=5)
            self.assertFalse(FitLock.is_held())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()

    def test_gpu_gate_counts_memory_and_fit_locks_only(self):
        with patch('src.proj.MACHINE', SimpleNamespace(cuda_server=True)), \
             patch.object(idle.sys, 'platform', 'linux'), patch.object(idle, 'gpu_memory_percent') as gpu:
            gpu.return_value = 19.99
            self.assertIsNone(idle.idle_reason({}))
            gpu.return_value = 20
            self.assertIn('busy', idle.idle_reason({}))
            with FitLock.guard():
                self.assertEqual(idle.idle_reason({}), 'fit lock occupied')
            gpu.side_effect = ValueError('no GPU')
            self.assertIn('unavailable', idle.idle_reason({}))

    def test_gpu_visible_uuid_and_unavailable_device(self):
        output = SimpleNamespace(stdout='0, GPU-a, 200, 1000\n1, GPU-b, 50, 1000\n')
        with patch.object(idle.subprocess, 'run', return_value=output), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-b'}):
            self.assertEqual(idle.gpu_memory_percent(), 5)
        with patch.object(idle.subprocess, 'run', return_value=output), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}):
            with self.assertRaises(ValueError):
                idle.gpu_memory_percent()

    def test_installer_creates_independent_unbounded_worker_unit(self):
        config = load_config(PATH.main / 'runs/scheduling', 'mengkjin-server')
        plan = build_plan(config, '', PATH.main, Path(sys.executable), 'Asia/Shanghai')
        worker = plan['units']['learndl-idle-worklist.service']
        validate_unit('learndl-idle-worklist.service', worker)
        self.assertNotIn('MemoryMax', worker)
        self.assertIn('TimeoutStartSec=infinity', worker)
        self.assertIn('MemoryMax=1G', plan['units']['learndl-watchdog.service'])
        self.assertEqual(config['watchdog']['jobs']['idle_worklist']['interval_seconds'], 900)


if __name__ == '__main__':
    unittest.main()
