"""Regression cases for version-patch diagnostics and interface lifecycle."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from typing import cast
from unittest.mock import Mock, patch

from src.automation import cli_recovery, patch_diagnostics, patch_services
from src.automation.scheduling import git_update
from src.automation.version_patch import AttemptRecord, Patch, _matches_success
from src.shared.file_ops import atomic_json


class PatchDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.log = self.root / 'patch.log'
        self.backup = self.root / 'backup'
        self.attempt = cast(AttemptRecord, dict(id='attempt', patch='0.7.2/001', commit_hash='head', host='host',
                            attempt=1, status='failed', error='Patch process exited with status 1',
                            log_path=str(self.log)))

    def test_report_precedes_generic_traceback_and_mail_retries(self):
        self.log.write_text('Version patch step: apply\nValueError: see report\n')
        atomic_json(self.backup / 'experiment-results.json', {'moves': [
            dict(state='blocked', error='Different files', source='old', target='new'),
            dict(state='blocked', error='Open files', source='other', target='other-new'),
        ]})
        summary = patch_diagnostics.failure_summary(self.attempt, persist=True)
        self.assertEqual((summary['step'], summary['error'], summary['blocked_count']), ('apply', 'Different files', 2))
        self.assertEqual(json.loads((self.root / 'failure.json').read_text()), summary)

        sender = Mock(side_effect=[False, True])
        with patch.object(git_update, 'runtime_dir', return_value=self.root):
            git_update.queue_version_failure(self.attempt)
            git_update.queue_version_failure(self.attempt)
            self.assertEqual(git_update.version_failure_mail_status('attempt'), 'queued')
            self.assertFalse(git_update.deliver_update_emails(sender))
            self.assertEqual(git_update.version_failure_mail_status('attempt'), 'waiting for retry')
            self.assertTrue(git_update.deliver_update_emails(sender))
            self.assertEqual(git_update.version_failure_mail_status('attempt'), 'SMTP accepted')
            self.assertTrue(git_update.deliver_update_emails(sender))
        self.assertEqual(sender.call_count, 2)
        for call in sender.call_args_list:
            self.assertIn('Different files', call.args[1])
            self.assertIn('source: old', call.args[1])
            self.assertEqual(call.kwargs['delivery_id'], 'git-update:version-attempt')

    def test_log_fallback_before_first_report(self):
        self.log.write_text('Version patch step: verify\nAssertionError: missing migrated output\n')
        result = patch_diagnostics.failure_summary(self.attempt)
        self.assertEqual(result['step'], 'verify')
        self.assertEqual(result['error'], 'AssertionError: missing migrated output')

    def test_failed_probe_reports_uncertainty(self):
        with patch.object(patch_diagnostics.subprocess, 'run', side_effect=FileNotFoundError('lsof')):
            with self.assertRaisesRegex(RuntimeError, 'Could not check open files'):
                patch_diagnostics.check_open_files(self.root)
        probe = subprocess.CompletedProcess([], 1, '', 'permission denied')
        with patch.object(patch_diagnostics.subprocess, 'run', return_value=probe):
            with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                patch_diagnostics.check_open_files(self.root)

    def test_unrelated_desktop_mount_warnings_do_not_block_experiment_migration(self):
        from src.automation.artifact_exchange.core import Config
        from src.automation.version_patch import load_patch_support

        layout = load_patch_support('0.7.2', 'result_layout')
        source = self.root / 'snapshot/tensorboard'
        target = self.root / 'results/tensorboard'
        source.mkdir(parents=True)
        (source / 'events.tfevents.1').write_text('events')
        warnings = ("lsof: WARNING: can't stat() fuse.portal file system /root/.cache/doc\n"
                    "  Output information may be incomplete.\n"
                    "lsof: WARNING: can't stat() fuse.gvfsd-fuse file system /root/.cache/gvfs\n"
                    "  Output information may be incomplete.\n")
        probe = subprocess.CompletedProcess([], 1, '', warnings)
        with (patch.object(layout, 'inventory', return_value=([layout.Move(source, target, 'tensorboard')], [])),
              patch.object(patch_diagnostics.subprocess, 'run', return_value=probe)):
            config = Config(role='publisher')
            summary = layout.run(config, self.root, self.root, self.root,
                                 report=self.backup / 'experiment-results.json')
        self.assertEqual(summary['moves'][0]['state'], 'migrated')
        self.assertEqual((target / 'events.tfevents.1').read_text(), 'events')

    def test_desktop_warnings_cannot_hide_open_files_or_probe_errors(self):
        unrelated = ("lsof: WARNING: can't stat() fuse.portal file system /root/.cache/doc\n"
                     "  Output information may be incomplete.\n")
        for name, probe, message in (
            ('open', subprocess.CompletedProcess([], 0, f'p42\ncpython\nn{self.log}\n', unrelated), 'PID 42'),
            ('error', subprocess.CompletedProcess([], 1, '', unrelated + 'lsof: permission denied\n'), 'permission denied'),
            ('parent', subprocess.CompletedProcess([], 1, '', unrelated.replace('/root/.cache/doc', str(self.root.parent))),
             'Could not check open files'),
            ('child', subprocess.CompletedProcess([], 1, '', unrelated.replace('/root/.cache/doc', str(self.root / 'mounted'))),
             'Could not check open files'),
            ('exit', subprocess.CompletedProcess([], 2, '', unrelated), 'lsof exited 2'),
        ):
            with self.subTest(name=name), patch.object(patch_diagnostics.subprocess, 'run', return_value=probe):
                with self.assertRaisesRegex(RuntimeError, message):
                    patch_diagnostics.check_open_files(self.root)

    def test_residual_patch_accepts_the_previous_successful_digest(self):
        from src.automation.version_patch import ROOT, discover

        entry = discover(ROOT, '0.7.7')['0.7.3/001']
        prior = '81ba2996a6f47dfe6329a896fed6e5d7bcb2c9274c7816e378461ae6d6fcd5b1'
        self.assertTrue(_matches_success(entry, entry.relative_path, prior))
        self.assertFalse(_matches_success(entry, entry.relative_path, 'unknown'))

    @unittest.skipUnless(shutil.which('lsof'), 'lsof required')
    def test_probe_identifies_actual_open_file(self):
        with self.log.open('w'):
            with self.assertRaisesRegex(RuntimeError, r'Open files at .*PID \d+'):
                patch_diagnostics.check_open_files(self.log)
        patch_diagnostics.check_open_files(self.log)

    def test_only_the_known_status_logger_revision_is_compatible(self):
        name = 'scripts/.version_patch/0.7.7/001_status_logger.py'
        known = '8fcdcac45e7b16adf9c0eeefa5f9f097b50f5787cb5e4256d053ba6ce2cb2020'
        current = 'b68b6b3bec58dbae2541085f28dc8a532c1c237cd8049e64b7628e2644ce8c93'
        self.assertTrue(_matches_success(Patch('0.7.7/001', '0.7.7', 1, name, current, 10), name, known))
        self.assertFalse(_matches_success(Patch('0.7.7/001', '0.7.7', 1, name, 'changed', 10), name, known))


class InterfaceLifecycleTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.services = patch_services.PatchServices(self.root, self.root / 'runtime')
        self.services.manager = Mock()
        self.instances = Mock()
        self.services.instances = self.instances
        self.services.manager.discover.return_value = []
        self.source = self.root / 'old.zip'
        self.target = self.root / 'new.zip'
        self.target.touch()
        self.instances.state = self.services.state / 'experiments'
        self.instances.discover.return_value = []
        self.instances.start_for_maintenance.return_value = {'url': 'ready'}
        dashboard = dict(pid=40, created=2, kind='optuna', source='old-cache', archive_source=str(self.source), port=8081)
        self.snapshot = dict(phase='migrating', restored=[], mappings={},
                             services={key: dict(pid=pid, created=1, port=port)
                                       for key, pid, port in [('tasks', 11, 8502), ('experiments', 12, 8505), ('console', 13, 8500)]},
                             experiments=[dashboard])
        self.scan = self.enterContext(patch.object(self.services, '_snapshot', return_value=self.snapshot))
        self.stop = self.enterContext(patch.object(patch_services, 'stop_process'))
        self.console = self.enterContext(patch.object(self.services, '_start_console'))
        self.patch_console = self.enterContext(patch.object(self.services, '_console', return_value={}))

    def test_failure_diagnostics_and_restart_from_original_journal(self):
        def check_saved(record, **kwargs):
            self.assertTrue(self.services.journal.exists())
            self.assertTrue(patch_services.interfaces_paused(self.root, 'tasks'))

        self.stop.side_effect = check_saved
        self.services.pause()
        self.assertEqual([call.args[0]['pid'] for call in self.stop.call_args_list], [40, 12, 11, 13])
        with self.assertRaisesRegex(ValueError, 'paused'):
            with patch_services.interface_operation(self.services.state, key='experiments'):
                self.fail('launch should be paused')
        blocked = self.services.restore(ready=False)
        self.assertEqual(blocked.get('restored'), ['console', 'tasks'])
        self.assertFalse(patch_services.interfaces_paused(self.root, 'tasks'))
        self.assertTrue(patch_services.interfaces_paused(self.root, 'experiments'))
        self.services.pause()
        self.assertEqual(self.scan.call_count, 1)
        backup = self.root / 'backup'
        atomic_json(backup / 'experiment-results.json', {'moves': [
            dict(state='migrated', source=str(self.source), target=str(self.target)),
        ]})
        self.services.collect_paths(backup)
        finished = self.services.restore(ready=True)
        self.assertEqual(finished.get('restored'), ['console', 'tasks', 'experiments', 'optuna:40'])
        self.instances.start_for_maintenance.assert_called_once_with('optuna', str(self.target), port=8081)
        self.assertFalse(self.services.journal.exists())

    def test_service_start_failure_is_retryable_without_another_migration(self):
        self.source.touch()
        self.services.pause()
        self.console.side_effect = [RuntimeError('port is busy'), None]
        failed = self.services.restore(ready=True)
        self.assertIn('port is busy', failed.get('errors', {}).get('console', ''))
        restored = self.services.restore(ready=True)
        self.assertFalse(restored.get('errors'))
        self.instances.start_for_maintenance.assert_called_once()
        self.assertFalse(self.services.journal.exists())

    def test_desktop_recovery_does_not_restart_cli_during_migration(self):
        self.services.pause()
        path = self.services.cli_path
        atomic_json(path, {'root': str(self.root), 'desired': True})
        with patch.object(cli_recovery.sys, 'platform', 'linux'), patch.object(cli_recovery, 'dispatch') as dispatch:
            self.assertEqual(cli_recovery.tick(path)['status'], 'version-maintenance')
            dispatch.assert_not_called()
