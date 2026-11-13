"""Delivery failures must never remove the only received copy."""
import os
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import cast
from unittest.mock import patch
from src.automation.artifact_exchange import ArtifactSpec, Config, publish_artifacts, receive
from src.automation.artifact_exchange.core import checked_entries, retry, locked
from src.automation.artifact_exchange.management import migration, service
from src.automation.artifact_exchange.research import analysis_specs
from src.api.learndl_monitor.experiment_archives import unpack_experiment


class ExchangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        shared = self.root/'shared'
        shared.mkdir()
        (shared/'.stfolder').mkdir()
        self.pub = Config(True, 'publisher', shared, self.root/'publisher', self.root/'server_results')
        self.rec = Config(True, 'receiver', shared, self.root/'receiver', self.root/'results')
        self.source = self.root/'source'
        self.source.mkdir()
        (self.source/'comparison.pdf').write_bytes(b'%PDF sample')
        (self.source/'comparison.xlsx').write_bytes(b'xlsx sample')

    def publish(self, spec=None, run='run'):
        rows = publish_artifacts('research', run, [spec or ArtifactSpec(self.source)], config=self.pub)
        self.assertEqual(rows[0]['status'], 'published', rows)
        return next(self.pub.inbox.rglob('*.transfer.zip'))

    def test_report_round_trip_idempotency_and_original_retained(self):
        archive = self.publish()
        data = archive.read_bytes()
        result = receive(self.rec)[0]
        self.assertEqual(result['status'], 'received')
        target = Path(result['target'])
        self.assertEqual((target/'results/comparison.pdf').read_bytes(), b'%PDF sample')
        self.assertTrue((self.source/'comparison.pdf').exists())
        self.assertFalse(archive.exists())
        archive.write_bytes(data)
        self.assertEqual(receive(self.rec)[0]['target'], str(target))
        self.assertFalse(archive.exists())
        self.assertEqual(publish_artifacts('research', 'run', [ArtifactSpec(self.source)], config=self.pub)[0]['status'], 'published')
        self.assertFalse(archive.exists())

    def test_experiment_round_trip_and_dashboard_import(self):
        events = self.root/'events'
        events.mkdir()
        (events/'events.out.tfevents.1').write_bytes(b'events')
        db = self.root/'study.sqlite3'
        with sqlite3.connect(db) as con:
            con.execute('create table results (x integer)')
            con.execute('insert into results values (42)')
        for kind, path in [('tensorboard', events), ('optuna', db)]:
            self.publish(ArtifactSpec(path, kind=kind), run=kind)
            result = receive(self.rec)[0]
            archive = next(Path(result['target']).rglob('*.zip'))
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.rec):
                source = unpack_experiment(kind, archive, self.root/'imports')
            self.assertTrue(source.exists())

    def test_failed_publication_keeps_snapshot_for_retry(self):
        (self.pub.shared_root/'.stfolder').rmdir()
        rows = publish_artifacts('test', 'run', [ArtifactSpec(self.source)], config=self.pub)
        self.assertEqual(rows[0]['status'], 'failed')
        (self.source/'comparison.pdf').write_bytes(b'changed after snapshot')
        (self.pub.shared_root/'.stfolder').mkdir()
        self.assertEqual(retry(self.pub)[0]['status'], 'published')
        target = Path(receive(self.rec)[0]['target'])
        self.assertEqual((target/'results/comparison.pdf').read_bytes(), b'%PDF sample')

    def test_crash_after_target_commit_before_receipt(self):
        archive = self.publish()
        from src.automation.artifact_exchange import core
        original = core.atomic_json
        def crash(path, value):
            if Path(path).parent.name == 'received':
                raise OSError('simulated disk full')
            return original(path, value)
        with patch.object(core, 'atomic_json', side_effect=crash):
            self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        self.assertTrue(archive.exists())
        self.assertEqual(receive(self.rec)[0]['status'], 'received')

    def test_target_conflict_keeps_package(self):
        archive = self.publish()
        data = archive.read_bytes()
        target = Path(receive(self.rec)[0]['target'])
        (target/'results/comparison.pdf').write_bytes(b'user edit')
        archive.write_bytes(data)
        self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        self.assertTrue(archive.exists())

    def test_unsafe_zip_names_and_limits(self):
        for names in [['../evil'], ['/absolute'], ['a', 'A'], ['a', 'a/b'], ['a\\b']]:
            archive = self.root/'invalid.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for n in names:
                    z.writestr(n, b'123')
            with zipfile.ZipFile(archive) as z, self.assertRaises(ValueError):
                checked_entries(z, self.rec)
        archive = self.publish()
        self.rec.max_bytes = 1
        self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        self.assertTrue(archive.exists())

    def test_tamper_duplicate_id_and_partial_file(self):
        archive = self.publish()
        with zipfile.ZipFile(archive) as z:
            members = {i.filename: z.read(i) for i in z.infolist()}
        members['payload/results/comparison.pdf'] = b'bad'
        with zipfile.ZipFile(archive, 'w') as z:
            for n, data in members.items():
                z.writestr(n, data)
        self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        archive.rename(archive.with_suffix('.partial'))
        self.assertEqual(receive(self.rec), [])

    def test_symlinks_disabled_and_sandbox(self):
        (self.source/'link').symlink_to(self.source/'comparison.pdf')
        self.assertEqual(publish_artifacts('test', 'run', [ArtifactSpec(self.source)], config=self.pub)[0]['status'], 'failed')
        with patch.dict(os.environ, {'LEARNDL_AUTOFIX_ID': 'test'}):
            self.assertEqual(publish_artifacts('test', 'run', [], config=self.pub)[0]['reason'], 'autofix')
            self.assertEqual(receive(self.rec), [])
        self.pub.enabled = False
        self.assertEqual(publish_artifacts('test', 'run', [], config=self.pub)[0]['status'], 'skipped')

    def test_lock_and_migration_rollback(self):
        with locked(self.rec, 'receive'), self.assertRaises(BlockingIOError):
            receive(self.rec)
        old = self.rec.results/'model_compare'
        old.mkdir(parents=True)
        report = old/'report.pdf'
        report.write_bytes(b'old')
        os.utime(report, (1, 1))
        rows = migration('plan', self.rec)
        self.assertEqual(next(r for r in rows if r['source'] == str(old))['status'], 'ready')
        migration('apply', self.rec)
        self.assertTrue(old.is_symlink())
        self.assertEqual(report.read_bytes(), b'old')
        migration('rollback', self.rec)
        self.assertFalse(old.is_symlink())
        self.assertEqual(report.read_bytes(), b'old')

    def test_bundle_handoff_excludes_weights_and_preserves_logs(self):
        from src.automation.artifact_exchange.research import handoff_bundle
        from src.res.rl_experimental.bundle import create_bundle, bundle_attachments
        (self.source/'final_model.zip').write_bytes(b'weights')
        (self.source/'stdout.log').write_text('research log')
        metadata = create_bundle(self.source, self.root/'bundles')
        legacy = bundle_attachments(metadata)
        with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub):
            records = handoff_bundle(metadata, self.source, 'rl')
        self.assertTrue(all(r['status'] == 'published' for r in records))
        attachments = bundle_attachments(metadata)
        self.assertIn(self.source/'stdout.log', attachments)
        self.assertFalse(any(p in legacy for p in attachments))
        with patch.dict(os.environ, {'LEARNDL_AUTOFIX_ID': 'test'}):
            self.assertEqual(bundle_attachments(metadata), legacy)
        received = receive(self.rec)
        self.assertTrue(received)
        for r in received:
            self.assertFalse(list(Path(r['target']).rglob('final_model.zip')))

    def test_same_package_id_different_archive_is_rejected(self):
        import json
        archive = self.publish()
        with zipfile.ZipFile(archive) as z:
            items = {i.filename: z.read(i) for i in z.infolist()}
        receive(self.rec)
        manifest = json.loads(items['manifest.json'])
        manifest['status'] = 'changed'
        items['manifest.json'] = json.dumps(manifest).encode()
        with zipfile.ZipFile(archive, 'w') as z:
            for name, content in items.items():
                z.writestr(name, content)
        self.assertEqual(receive(self.rec)[0]['status'], 'failed')
        self.assertTrue(archive.exists())

    def test_batch_all_roles_hosts_models_and_panel(self):
        from dataclasses import replace
        from src.api.learndl_monitor.experiments import synced_sources
        events = self.root/'events'
        events.mkdir()
        (events/'events.out.tfevents.1').write_bytes(b'events')
        db = self.root/'study.sqlite3'
        with sqlite3.connect(db) as con:
            con.execute('create table results(x integer)')
            con.execute('insert into results values (42)')
        (self.source/'full.feather').write_bytes(b'arbitrary completed result')
        (self.source/'report.zip').write_bytes(b'not an experiment')
        specs = [ArtifactSpec(self.source, category='model_training', experiment='nn@modelAAA'),
                 ArtifactSpec(events, kind='tensorboard', category='model_training', experiment='nn@modelAAA'),
                 ArtifactSpec(db, kind='optuna', category='model_training', experiment='nn@modelAAA')]
        self.pub.source_host = 'server'
        records = publish_artifacts('train', 'run1', specs, config=self.pub)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'published', records)
        self.assertEqual(records[0]['manifest']['schema_version'], 2)
        self.pub.source_host = 'laptop'
        publish_artifacts('train', 'run1', specs + [replace(specs[0], experiment='boost@modelAAA')], config=self.pub)
        received = receive(self.rec)
        self.assertEqual(len(received), 3)
        self.assertTrue(all(r['status'] == 'received' for r in received), received)
        first = Path(next(r['target'] for r in received if 'server__' in r['target']))
        self.assertEqual(first.parent.name, 'server__nn@modelAAA')
        self.assertTrue((first/'results/full.feather').exists())
        self.assertTrue((first/'results/report.zip').exists())
        sources = synced_sources(self.rec.results)
        self.assertEqual(len(sources['tensorboard']), 2)
        self.assertEqual(len(sources['optuna']), 2)
        for kind, items in sources.items():
            for item in items:
                with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.rec):
                    imported = unpack_experiment(kind, Path(item['path']), self.root/'imports')
                self.assertTrue(imported.exists())
        # A staging tree with the same manifest is not a completed batch.
        import shutil
        shutil.copytree(first, first.parent/'.receive-incomplete')
        self.assertEqual(synced_sources(self.rec.results), sources)

    def test_legacy_v1_packet_and_migration_alias(self):
        import json
        import hashlib
        from src.automation.artifact_exchange.core import _target
        pid = 'a' * 32
        data = b'legacy report'
        manifest = dict(schema_version=1, id=pid, task='old', run_id='run',
                        batch='run__'+pid, kind='reports', category='model_compare',
                        files=[dict(path='report.txt', size=len(data), sha256=hashlib.sha256(data).hexdigest())])
        archive = self.pub.inbox/'reports/model_compare/legacy.transfer.zip'
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('manifest.json', json.dumps(manifest))
            z.writestr('payload/report.txt', data)
        received = receive(self.rec)[0]
        self.assertEqual(received['status'], 'received', received)
        self.assertEqual(Path(received['target']), _target(manifest, self.rec))
        legacy = self.rec.results/'reports/model_compare/legacy/model_compare'
        legacy.mkdir(parents=True)
        (legacy/'old.pdf').write_bytes(b'old')
        (self.rec.results/'model_compare').symlink_to(legacy)
        self.publish(ArtifactSpec(self.source, category='model_compare', experiment='compare'))
        received = receive(self.rec)[0]
        self.assertEqual(received['status'], 'received', received)
        self.assertTrue((legacy/'old.pdf').exists())

    def test_destination_conflicts_custom_category_and_stable_host(self):
        from src.automation.artifact_exchange.core import atomic_json
        config_path = self.root/'pub.json'
        atomic_json(config_path, {'enabled': True, 'role': 'publisher', 'shared_root': str(self.pub.shared_root),
                                 'state': str(self.pub.state), 'results': str(self.pub.results)})
        first = Config.load(config_path)
        first.source_host = 'saved-publisher'
        records = publish_artifacts('custom', 'run', [ArtifactSpec(self.source, category='new_research')], config=first)
        self.assertEqual(records[0]['status'], 'published')
        with patch('socket.gethostname', return_value='changed-wifi-name'):
            self.assertEqual(Config.load(config_path).source_host, first.source_host)
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        rows = publish_artifacts('task', 'collision', [ArtifactSpec(self.source),
            ArtifactSpec(self.source/'comparison.pdf')], config=self.pub)
        self.assertEqual(rows[0]['status'], 'failed')
        with self.assertRaises(ValueError):
            ArtifactSpec(self.source, destination='../outside')

    def test_model_scope_batches_full_results_after_writer_close(self):
        from types import SimpleNamespace
        from src.automation.artifact_exchange.research import model_run, register, CURRENT_TASK
        reports = self.source
        events = self.root/'events'
        events.mkdir()
        (events/'events.out.tfevents.1').write_bytes(b'event')
        calls = []
        def close():
            calls.append('closed')
            (reports/'complete.binary').write_bytes(b'all files')
        trainer = SimpleNamespace(callback=SimpleNamespace(callbacks=[SimpleNamespace(reset_writer=close)]),
            config=SimpleNamespace(base_path=SimpleNamespace(full_name='nn@modelAAA',
                rslt=lambda: reports, snapshot=lambda name: events)))
        owner = SimpleNamespace(_artifact_results=[], _artifacts=[])
        token = CURRENT_TASK.set(owner)
        try:
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub), \
                    patch('src.proj.Save.async_wait_all') as wait:
                with model_run(trainer):
                    self.assertFalse(list(self.pub.inbox.rglob('*.transfer.zip')))
                wait.assert_called_once()
            self.assertEqual(calls, ['closed'])
            self.assertEqual(len(owner._artifact_results), 1)
            self.assertEqual(owner._artifact_results[0]['status'], 'prepared')
            self.assertEqual(receive(self.rec), [])
            (reports/'complete.binary').write_bytes(b'next model overwrote this file')
            from src.automation.artifact_exchange.research import finalize_task_artifacts
            finalize_task_artifacts(owner)
            received = receive(self.rec)[0]
            self.assertEqual((Path(received['target'])/'results/complete.binary').read_bytes(), b'all files')
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub), \
                    patch('src.proj.Save.async_wait_all', side_effect=OSError('pending saves')):
                with self.assertRaisesRegex(RuntimeError, 'research failed'):
                    with model_run(trainer):
                        register(ArtifactSpec(reports), 'test', 'run')
                        raise RuntimeError('research failed')
            self.assertEqual(len(owner._artifact_results), 1)
        finally:
            CURRENT_TASK.reset(token)

    def test_snapshot_detects_late_changes_and_migration_skips_batches(self):
        from src.automation.artifact_exchange import core
        original = core._copy
        def changing(source, target):
            original(source, target)
            if source.name == 'comparison.xlsx':
                (self.source/'comparison.pdf').write_bytes(b'late modification')
        with patch.object(core, '_copy', side_effect=changing):
            rows = publish_artifacts('test', 'changing', [ArtifactSpec(self.source)], config=self.pub)
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertFalse(list(self.pub.inbox.rglob('*.transfer.zip')))
        self.publish(ArtifactSpec(self.source, category='model_compare'))
        self.assertEqual(receive(self.rec)[0]['status'], 'received')
        rows = migration('plan', self.rec)
        row = next(r for r in rows if r['source'] == str(self.rec.results/'model_compare'))
        self.assertIn('Contains received experiment batches', row['error'])

    def test_task_context_resets_even_when_cleanup_fails(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.proj.util.script.autorun import AutoRunTask
        from src.automation.artifact_exchange.research import CURRENT_TASK
        outer = object()
        outer_token = CURRENT_TASK.set(outer)
        inner = SimpleNamespace(_finish_task=Mock(side_effect=OSError('cleanup failed')))
        inner._artifact_token = CURRENT_TASK.set(inner)
        try:
            with self.assertRaisesRegex(OSError, 'cleanup failed'):
                AutoRunTask.__exit__(cast(AutoRunTask, inner), None, None, None)
            self.assertIs(CURRENT_TASK.get(), outer)
        finally:
            CURRENT_TASK.reset(outer_token)

    def test_mail_handoff_checks_current_bytes_and_keeps_logs(self):
        from src.automation.artifact_exchange.research import delivered_report_paths
        (self.source/'run.log').write_text('log')
        (self.source/'execution.html').write_text('execution report')
        records = publish_artifacts('research', 'mail', [ArtifactSpec(self.source)], config=self.pub)
        handed = delivered_report_paths(records)
        self.assertEqual(handed, {str(self.source/'comparison.pdf'), str(self.source/'comparison.xlsx')})
        (self.source/'comparison.pdf').write_text('new untransferred report')
        self.assertEqual(delivered_report_paths(records), {str(self.source/'comparison.xlsx')})
        self.assertEqual(delivered_report_paths([dict(records[0], status='failed')]), set())

    def test_queued_delivery_deduplicates_and_retries_frozen_snapshot(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.automation.artifact_exchange.research import CURRENT_TASK, publish_safe, finalize_task_artifacts
        owner = SimpleNamespace(_artifacts=[], _artifact_results=[], logger=Mock())
        token = CURRENT_TASK.set(owner)
        try:
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub):
                first = publish_safe('test', 'frozen', [ArtifactSpec(self.source)])
                again = publish_safe('test', 'frozen', [ArtifactSpec(self.source)])
            self.assertIs(first[0], again[0])
            self.assertEqual(len(owner._artifact_results), 1)
            (self.source/'comparison.pdf').write_bytes(b'new result')
            (self.pub.shared_root/'.stfolder').rmdir()
            finalize_task_artifacts(owner)
            self.assertEqual(first[0]['status'], 'failed')
            self.assertTrue(owner.logger.warning.called)
            (self.pub.shared_root/'.stfolder').mkdir()
            self.assertEqual(retry(self.pub)[0]['status'], 'published')
            target = Path(receive(self.rec)[0]['target'])
            self.assertEqual((target/'results/comparison.pdf').read_bytes(), b'%PDF sample')
        finally:
            CURRENT_TASK.reset(token)

    def test_task_delivery_disabled_and_autofix_are_logged_without_snapshots(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.automation.artifact_exchange.research import CURRENT_TASK, publish_safe, finalize_task_artifacts
        for autofix in (False, True):
            owner = SimpleNamespace(_artifacts=[], _artifact_results=[], logger=Mock())
            token = CURRENT_TASK.set(owner)
            self.pub.enabled = False
            try:
                with patch.dict(os.environ, {'LEARNDL_AUTOFIX_ID': 'test' if autofix else ''}), \
                        patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub) as load, \
                        patch('src.automation.artifact_exchange.core._snapshot') as snapshot:
                    rows = publish_safe('test', 'disabled', [ArtifactSpec(self.source)])
                    finalize_task_artifacts(owner)
                    self.assertEqual(rows[0]['status'], 'skipped')
                    snapshot.assert_not_called()
                    self.assertTrue(owner.logger.note.called)
                    if autofix:
                        load.assert_not_called()
            finally:
                CURRENT_TASK.reset(token)

    def test_task_logs_delivery_before_capture_close_and_mail(self):
        from datetime import datetime
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.env.variable.files import UniqueFileList
        from src.automation.artifact_exchange.research import CURRENT_TASK, defer_mail
        import src.proj.util.script.autorun as autorun
        for fail in (False, True):
            with self.subTest(fail=fail):
                events = []
                captured = []
                capture_open = [True]
                def log(message, **kwargs):
                    self.assertTrue(capture_open[0], 'Delivery status must be captured')
                    captured.append(message)
                    events.append('log')
                def close(*args):
                    events.append('close')
                    capture_open[0] = False
                owner = SimpleNamespace(_artifact_results=[], _artifacts=[ArtifactSpec(self.source)],
                    task_name='order-test', init_time=datetime.now(), execution_status='Success',
                    should_forfeit=False, func_return=None, error_messages=[], execution_success=True,
                    source='test', task_recorder=Mock(), status='Success',
                    catchers=SimpleNamespace(exit=close),
                    logger=SimpleNamespace(note=log, warning=log, error=log,
                        get_conclusions=lambda level: [], draw_conclusions=lambda: 'summary'))
                def mail():
                    events.append('mail')
                    self.assertFalse(capture_open[0])
                    self.assertTrue(any(('failed' if fail else 'published') in m for m in captured))
                    self.assertEqual(owner.execution_status, 'Success')
                owner.send_email = mail
                files = UniqueFileList('test_order_exit_'+str(fail))
                files.append(self.source/'comparison.pdf')
                fake_proj = SimpleNamespace(exit_files=files, email_attachments=UniqueFileList('test_order_mail_'+str(fail)))
                if fail:
                    (self.pub.shared_root/'.stfolder').rmdir()
                token = CURRENT_TASK.set(owner)
                try:
                    defer_mail(lambda: events.append('deferred_mail') or {'status': 'sent'})
                    with patch.object(autorun, 'Proj', fake_proj), patch.object(autorun.Save, 'async_wait_all'), \
                            patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub):
                        AutoRunTask._finish_task(cast(AutoRunTask, owner), None, None, None)
                    self.assertLess(events.index('log'), events.index('close'))
                    self.assertLess(events.index('close'), events.index('deferred_mail'))
                    self.assertLess(events.index('deferred_mail'), events.index('mail'))
                    self.assertEqual(bool(owner.exit_files), fail)
                finally:
                    CURRENT_TASK.reset(token)
                    (self.pub.shared_root/'.stfolder').mkdir(exist_ok=True)

    def test_training_mail_waits_for_task_publication(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.automation.artifact_exchange.research import CURRENT_TASK, finalize_task_artifacts, run_deferred_mail
        from src.data.preprocess.factor_sequence.training_mail import send_training_result
        owner = SimpleNamespace(_artifact_results=[], _artifacts=[], logger=Mock())
        token = CURRENT_TASK.set(owner)
        try:
            result = dict(name='attempt', model='test-model', seed=1, status='passed')
            log = self.source/'attempt.log'
            log.write_text('training output')
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub), \
                    patch('src.proj.MACHINE', SimpleNamespace(emailable=True)), \
                    patch('src.proj.util.web.task_mail.TaskResultMail') as sender:
                sender.return_value.return_value = True
                state = send_training_result(self.source, result)
                self.assertEqual(state['status'], 'queued')
                sender.assert_not_called()
                self.assertFalse(list(self.pub.inbox.rglob('*.transfer.zip')))
                finalize_task_artifacts(owner)
                sender.assert_not_called()
                self.assertEqual(owner._artifact_results[0]['status'], 'published')
                run_deferred_mail(owner)
                self.assertEqual(state['status'], 'sent')
                attachments = sender.return_value.call_args.kwargs['attachments']
                self.assertEqual(attachments, [log])
                self.assertIn('published', sender.return_value.call_args.args[1])
        finally:
            CURRENT_TASK.reset(token)

    def test_task_bundle_replaces_queued_parts_only_after_success(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from src.proj.env.variable.files import UniqueFileList
        from src.res.rl_experimental.bundle import create_bundle, bundle_attachments
        from src.automation.artifact_exchange.research import CURRENT_TASK, finalize_task_artifacts, handoff_bundle
        metadata = create_bundle(self.source, self.root/'bundles')
        originals = bundle_attachments(metadata)
        files = UniqueFileList('test_bundle_queued')
        for p in originals:
            files.append(p)
        owner = SimpleNamespace(_artifact_results=[], _artifacts=[], logger=Mock())
        token = CURRENT_TASK.set(owner)
        try:
            with patch('src.automation.artifact_exchange.core.Config.load', return_value=self.pub), \
                    patch('src.proj.Proj', SimpleNamespace(email_attachments=files, exit_files=UniqueFileList('test_bundle_exit'))):
                records = handoff_bundle(metadata, self.source, 'rl')
                self.assertEqual(records[0]['status'], 'prepared')
                self.assertEqual(bundle_attachments(metadata), originals)
                finalize_task_artifacts(owner)
                self.assertEqual(records[0]['status'], 'published')
                attachments = [Path(p) for p in files.pop_all()]
                self.assertFalse(any(p in originals for p in attachments))
                self.assertIn(metadata.with_suffix('.artifacts.json'), attachments)
        finally:
            CURRENT_TASK.reset(token)

    def test_analysis_selection_and_service_plan(self):
        (self.source/'weights.pt').write_bytes(b'weights')
        (self.source/'final_model.zip').write_bytes(b'weights')
        specs = analysis_specs(self.source, 'rl')
        self.assertEqual(set(specs[0].members), {'comparison.pdf', 'comparison.xlsx'})
        definition = service('plan', self.root/'config with spaces.json', self.rec)['definition']
        self.assertEqual(definition['StartInterval'], 300)
        self.assertIn(str(self.root/'config with spaces.json'), definition['ProgramArguments'])


if __name__ == '__main__':
    unittest.main()
