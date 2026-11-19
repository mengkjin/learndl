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
from src.automation.version_autofix import load_patch_support

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
        models = self.root / 'models'
        args = (self.rec, models, models / 'archive', self.root / 'optuna')
        with patch.object(result_layout, '_idle'):
            result_layout.run(*args, report=self.root / 'moves.json')
            result_layout.run(*args, report=self.root / 'moves.json')
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
