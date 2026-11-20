from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.automation.artifact_exchange.core import ArtifactSpec, Config, atomic_json, publish_artifacts, receive, retry
from src.automation.version_patch import load_patch_support

mail_backfill = load_patch_support('0.7.2', 'mail_backfill')
result_layout = load_patch_support('0.7.2', 'result_layout')


class ArtifactUpgradeTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name).resolve()
        self.shared = self.root / 'share'
        self.shared.mkdir()
        self.pub = Config(True, 'publisher', self.shared, self.root / 'pub', self.root / 'server')
        self.rec = Config(True, 'receiver', self.shared, self.root / 'rec', self.root / 'client')
        self.outbox = self.root / 'email_outbox'

    def test_legacy_batch_is_resolved_then_migrated_without_manifest_change(self):
        from src.api.learndl_monitor.experiments import synced_sources

        self.shared.joinpath('.stfolder').mkdir()
        events = self.root / 'events'
        events.mkdir()
        (events / 'events.tfevents.1').write_bytes(b'event')
        spec = ArtifactSpec(events, kind='tensorboard', category='model_training', experiment='nn@test',
                            destination='tensorboard')
        publish_artifacts('old', 'first', [spec], config=self.pub)
        target = Path(receive(self.rec)[0]['target'])
        canonical, = target.glob('results/tensorboard/*.zip')
        legacy = target / 'tensorboard' / canonical.name
        legacy.parent.mkdir()
        canonical.rename(legacy)
        before = (target / 'source.json').read_bytes()
        self.assertEqual(synced_sources(self.rec.results)['tensorboard'][0]['path'], str(legacy))
        from src.automation.artifact_exchange.core import _target
        current = target
        target = _target(json.loads((current/'source.json').read_text()), self.rec)
        target.parent.mkdir(parents=True)
        current.rename(target)
        models = self.root / 'models'
        args = (self.rec, models, models / 'archive', self.root / 'optuna')
        with patch.object(result_layout, '_idle'):
            result_layout.run(*args, report=self.root / 'moves.json')
            result_layout.run(*args, report=self.root / 'moves.json')
        target.rename(current)
        target = current
        self.assertEqual((target / 'source.json').read_bytes(), before)
        self.assertTrue(canonical.is_file())
        self.assertFalse(legacy.exists())
        self.assertEqual(synced_sources(self.rec.results)['tensorboard'][0]['path'], str(canonical))

    def test_sqlite_backup_preserves_wal_and_conflict(self):
        source = self.root / 'original.sqlite3'
        target = self.root / 'results/optuna/copied.sqlite3'
        move = result_layout.Move(source, target, 'sqlite')
        with closing(sqlite3.connect(source)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('create table trials (number integer)')
            db.execute('insert into trials values (72)')
            db.commit()
            result_layout._apply(move, *result_layout._check(move, check_open=False))
            with closing(sqlite3.connect(target)) as copied:
                self.assertEqual(copied.execute('select number from trials').fetchall(), [(72,)])
            self.assertTrue(result_layout._check(move, check_open=False)[0])
            db.execute('insert into trials values (73)')
            db.commit()
            with self.assertRaises(FileExistsError):
                result_layout._check(move, check_open=False)
            self.assertTrue(source.is_file() and target.is_file())

    def queue_mail(self, name, date, subject, *, sent=True, eml=True):
        folder = self.outbox / '.archive' / name
        atomic_json(folder / 'meta.json', {'created_at': date.isoformat()})
        atomic_json(folder / 'payload/manifest.json', {'messages': [{'sent': sent}]})
        if eml:
            message = EmailMessage()
            message['Subject'] = subject
            message.set_content(name)
            message.add_attachment(name.encode(), maintype='application', subtype='octet-stream',
                                   filename='report.7z.001')
            (folder / 'message.eml').write_bytes(message.as_bytes())

    def test_backfill_fixed_window_verified_spool_and_replay(self):
        end = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
        self.queue_mail('boundary', end - timedelta(days=7), 'Success - Daily Update - 1')
        self.queue_mail('expired', end - timedelta(days=7, seconds=1), 'Success - Daily Update - 2')
        self.queue_mail('error', end, 'Error - Resume Testing - 3')
        self.queue_mail('missing', end, 'Success - Test Model - 4', eml=False)
        self.queue_mail('unsent', end, 'Success - Daily Update - 5', sent=False)
        self.queue_mail('unrelated', end, 'Success - Different Task - 6')
        preflight = mail_backfill.run(self.pub, self.outbox, now=end)
        self.assertEqual(preflight['counts']['matched'], 2)
        self.assertFalse(self.pub.state.exists())
        report = self.root / 'report.json'
        result = mail_backfill.run(self.pub, self.outbox, report=report, now=end)
        self.assertEqual(result['counts']['pending'], 2)
        self.shared.joinpath('.stfolder').mkdir()
        self.assertEqual(len(retry(self.pub)), 2)
        rerun = mail_backfill.run(self.pub, self.outbox, report=report, now=end + timedelta(days=9))
        self.assertEqual(rerun['end'], end.isoformat())
        self.assertEqual(rerun['counts']['published'], 2)
        delivered = receive(self.rec)
        self.assertEqual(len(delivered), 2)
        metadata = [json.loads((Path(row['target']) / 'mail/mail.json').read_text()) for row in delivered]
        self.assertEqual({row['status'] for row in metadata}, {'Error', 'Success'})
        self.assertEqual({row['smtp_status'] for row in metadata}, {'sent'})
        self.assertEqual({row['attachments'][0]['name'] for row in metadata}, {'report.7z.001'})

    def test_factor_residual_patch_moves_local_archive_and_received_results(self):
        from runpy import run_path
        from types import SimpleNamespace
        from src.automation.artifact_exchange.core import _verify_target
        from src.automation.version_patch import ROOT

        self.shared.joinpath('.stfolder').mkdir()
        source = self.root / 'package/factor_residual'
        source.mkdir(parents=True)
        (source / '20261119.feather').write_bytes(b'residual')
        spec = ArtifactSpec(source.parent, category='model_training', experiment='nn@test')
        publish_artifacts('model_training', 'old', [spec], config=self.pub)
        batch = Path(receive(self.rec)[0]['target'])
        snapshot = batch / 'snapshot/factor_residual'
        self.assertEqual((snapshot / '20261119.feather').read_bytes(), b'residual')
        manifest = (batch / 'source.json').read_bytes()
        _verify_target(batch, json.loads(manifest))

        # Also emulate the layout of a batch received before updating.
        from src.automation.artifact_exchange.core import _target
        current = batch
        batch = _target(json.loads((current/'source.json').read_text()), self.rec)
        batch.parent.mkdir(parents=True)
        current.rename(batch)
        snapshot = batch / 'snapshot/factor_residual'
        legacy = batch / 'results/factor_residual'
        legacy.parent.mkdir()
        snapshot.rename(legacy)
        models = self.root / 'models'
        active = models / 'nn/model'
        archived = models / 'archive/model.20261119000000'
        for base in (active, archived):
            folder = base / 'results/factor_residual'
            folder.mkdir(parents=True)
            (folder / 'daily.csv').write_bytes(b'diagnostics')
        migration = run_path(str(ROOT / 'scripts/.version_patch/0.7.3/001_factor_residual_snapshot.py'))
        ctx = SimpleNamespace(root=self.root, backup_dir=self.root / 'patch-report')
        with patch('src.proj.PATH', SimpleNamespace(model=models)), patch.object(Config, 'load', return_value=self.rec), \
                patch('subprocess.run', return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'')):
            for _ in range(2):
                migration['apply'](ctx)
                self.assertTrue(migration['verify'](ctx))
        self.assertEqual((batch / 'source.json').read_bytes(), manifest)
        _verify_target(batch, json.loads(manifest))
        for base in (active, archived):
            self.assertFalse((base / 'results/factor_residual').exists())
            self.assertEqual((base / 'snapshot/factor_residual/daily.csv').read_bytes(), b'diagnostics')

    def test_results_workbook_preview_is_bounded_and_selects_sheet(self):
        from openpyxl import Workbook
        from src.api.learndl_monitor.artifact_library import Library

        book = Workbook()
        book.create_sheet('实验')
        sheet = book['实验']
        assert sheet is not None
        sheet['A1'] = '指标'
        sheet['B1'] = .25
        sheet['U1'] = 'outside visible columns'
        sheet['A41'] = 'outside visible rows'
        path = self.root / 'result.xlsx'
        book.save(path)
        catalog = Library('model_preview', self.rec, state=self.root / 'library')
        preview = catalog.spreadsheet({'type': 'file', 'path': path}, 0, '实验')
        self.assertEqual(preview, {'sheets': ['Sheet', '实验'], 'sheet': '实验', 'rows': [['指标', '0.25']]})
