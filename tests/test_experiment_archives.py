"""Round-trip naming, ZIP import safety and downloaded source discovery."""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.api.learndl_monitor.experiment_archives import unpack_experiment
from src.api.learndl_monitor.experiment_mail import experiment_name
from src.api.learndl_monitor.experiments import Instances

__all__ = []


class ExperimentArchivesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_full_model_names(self):
        for path, expected in [
            ('models/nn/gru@day@2/snapshot/tensorboard', 'nn@gru@day@2'),
            ('models/st/nn@gru@day/snapshot/tensorboard', 'st@nn@gru@day'),
            ('models/factor/df_scores/snapshot/tensorboard', 'factor@df_scores'),
            ('models/archive/nn@gru@day.20261112010203/snapshot/tensorboard', 'nn@gru@day'),
        ]:
            self.assertEqual(experiment_name('tensorboard', Path(path)), expected)
        self.assertEqual(experiment_name('optuna', Path('boost@lgbm@day@20261112010203.sqlite3')), 'boost@lgbm@day')

    def test_import_reuses_cache_and_preserves_model_name(self):
        archive = self.root / 'nn@gru@day-20261112-123000.zip'
        with zipfile.ZipFile(archive, 'w') as package:
            package.writestr('tensorboard/run/events.out.tfevents.1', 'logs')
        source = unpack_experiment('tensorboard', archive, self.root / 'cache')
        self.assertEqual(experiment_name('tensorboard', source), 'nn@gru@day')
        self.assertEqual(unpack_experiment('tensorboard', archive, self.root / 'cache'), source)
        self.assertEqual((source / 'tensorboard/run/events.out.tfevents.1').read_text(), 'logs')

    def test_reject_traversal_and_wrong_payload(self):
        for name in ['../escaped', '/absolute', 'other.txt']:
            archive = self.root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr(name, 'bad')
            with self.assertRaises(ValueError):
                unpack_experiment('tensorboard', archive, self.root / 'cache')
        self.assertFalse((self.root / 'escaped').exists())

    def test_downloads_are_categorized(self):
        from src.api.pkgs import dashboard

        for kind in ['tensorboard', 'optuna']:
            folder = self.root / 'results/model_experiments' / kind
            folder.mkdir(parents=True)
            (folder / 'model.zip').touch()
        paths = SimpleNamespace(model=self.root / 'models', model_archive=self.root / 'models/archive', optuna=self.root / 'optuna')
        with patch('src.proj.PATH', paths), patch('src.api.learndl_monitor.experiments.ROOT', self.root), patch.object(
                dashboard, 'archived_tensorboard_index', return_value={}):
            groups = Instances(self.root / 'state').sources()
        for kind in ['tensorboard', 'optuna']:
            self.assertEqual(groups[kind][-1]['items'][0]['path'], str(self.root / 'results/model_experiments' / kind / 'model.zip'))


if __name__ == '__main__':
    unittest.main()
