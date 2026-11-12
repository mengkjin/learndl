"""Isolated experiment export tests; never deliver real email."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.api.learndl_monitor.experiment_mail import send_experiment
from src.api.learndl_monitor.experiments import Instances

__all__ = []


class ExperimentMailTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.machine = patch('src.proj.MACHINE', SimpleNamespace(emailable=True))
        self.machine.start()
        self.addCleanup(self.machine.stop)

    def test_tensorboard_contains_only_source_and_no_symlink_targets(self):
        source = self.root / 'tensorboard'
        (source / 'run').mkdir(parents=True)
        (source / 'run/events.tfevents').write_bytes(b'event data')
        weights = self.root / 'weights.pt'
        weights.write_bytes(b'not included')
        (source / 'linked').symlink_to(weights)
        archives = []

        def deliver(*args, **kwargs):
            self.assertFalse(kwargs['project_attachments'])
            archives.extend(kwargs['attachments'])
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertEqual(archive.namelist(), ['tensorboard/run/events.tfevents'])
                self.assertEqual(archive.read(archive.namelist()[0]), b'event data')
            return True

        with patch('src.proj.util.web.emailer.Email.send', side_effect=deliver):
            self.assertTrue(send_experiment('tensorboard', source))
        self.assertFalse(archives[0].exists())

    def test_optuna_snapshot_includes_wal_and_only_database(self):
        source = self.root / 'study.sqlite3'
        connection = sqlite3.connect(source)
        self.addCleanup(connection.close)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE trial(value INTEGER)')
        connection.execute('INSERT INTO trial VALUES (42)')
        connection.commit()

        def deliver(*args, **kwargs):
            with zipfile.ZipFile(kwargs['attachments'][0]) as archive:
                self.assertEqual(archive.namelist(), ['study.sqlite3'])
                exported = self.root / 'export.sqlite3'
                exported.write_bytes(archive.read('study.sqlite3'))
                with sqlite3.connect(exported) as snapshot:
                    self.assertEqual(snapshot.execute('SELECT value FROM trial').fetchall(), [(42,)])
            return False

        with patch('src.proj.util.web.emailer.Email.send', side_effect=deliver):
            self.assertFalse(send_experiment('optuna', source))

    def test_email_uses_discovered_source_and_rejects_stale_identity(self):
        manager = Instances(self.root)
        row = dict(pid=12, created=34, kind='optuna', source='/discovered/study.sqlite3')
        request = dict(action='email', pid=12, created=34, kind='optuna', source='/untrusted')
        with patch.object(manager, 'discover', return_value=[row]), patch(
                'src.api.learndl_monitor.experiment_mail.send_experiment', return_value=True) as sender:
            self.assertEqual(manager.act(request), {'mail_status': 'sent'})
            sender.assert_called_once_with('optuna', Path('/discovered/study.sqlite3'))
            request['created'] = 99
            with self.assertRaisesRegex(ValueError, 'identity'):
                manager.act(request)
            self.assertEqual(sender.call_count, 1)

    def test_missing_mail_configuration_is_explicit(self):
        with patch('src.proj.MACHINE', SimpleNamespace(emailable=False)), patch(
                'src.proj.util.web.emailer.Email.send') as sender:
            with self.assertRaisesRegex(ValueError, 'not configured'):
                send_experiment('tensorboard', self.root)
            sender.assert_not_called()


if __name__ == '__main__':
    unittest.main()
