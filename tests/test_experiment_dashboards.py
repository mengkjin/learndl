"""Temporary-state tests for experiment instance limits and process identity."""
from __future__ import annotations

import tempfile
from types import SimpleNamespace
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.learndl_monitor.experiments import Instances, argument

__all__ = []


class ExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = Instances(self.root)

    def test_latest_training_uses_event_timestamp_not_copy_time(self):
        from src.api.pkgs import dashboard
        model = self.root / 'models'
        for name, stamp in [('older', 1600000000), ('newer', 1700000000)]:
            folder = model / name / 'snapshot/tensorboard'
            folder.mkdir(parents=True)
            event = folder / f'events.out.tfevents.{stamp}.host'
            event.write_text('')
            os.utime(event, (1800000000 if name == 'older' else 500, 1800000000 if name == 'older' else 500))
        paths = SimpleNamespace(model=model, model_archive=model / 'archive', optuna=self.root / 'optuna')
        with patch('src.proj.PATH', paths), patch.object(dashboard, '_tensorboard_snapshot_ready', return_value=True):
            groups = self.manager.sources()['tensorboard']
        self.assertEqual(groups[0]['en'], 'Latest training')
        self.assertEqual(len(groups[0]['items']), 1)
        self.assertIn('/newer/', groups[0]['items'][0]['path'])
        self.assertIn('2023-', groups[0]['items'][0]['label'])
        self.assertEqual(len(groups[1]['items']), 2)

    def test_arguments(self):
        self.assertEqual(argument(['--port=6009'], '--port'), '6009')
        self.assertEqual(argument(['--logdir', '/tmp/a b'], '--logdir'), '/tmp/a b')

    def test_limits_persist_and_reject_invalid_values(self):
        self.assertEqual(self.manager.settings(), {'tensorboard': 5, 'optuna': 5})
        self.manager.act({'kind': 'optuna', 'action': 'limit', 'value': 2})
        self.assertEqual(Instances(self.root).settings()['optuna'], 2)
        for value in (True, 0, -1, 1.2):
            with self.assertRaises(ValueError):
                self.manager.act({'kind': 'optuna', 'action': 'limit', 'value': value})

    def test_external_instances_count_and_duplicate_reuses(self):
        rows = [dict(kind='tensorboard', source='/other', pid=i, created=1) for i in range(5)]
        with patch.object(self.manager, 'discover', return_value=rows), patch('subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'limit'):
                self.manager.act({'action': 'start', 'kind': 'tensorboard', 'source': str(self.root)})
            rows[0]['source'] = str(self.root.resolve())
            result = self.manager.act({'action': 'start', 'kind': 'tensorboard', 'source': str(self.root)})
            self.assertTrue(result['reused'])
            launch.assert_not_called()

    def test_stopping_manager_keeps_instance_children(self):
        from src.api.learndl_monitor import services
        process = MagicMock(pid=123)
        process.create_time.return_value = 456
        with patch.object(services, 'STATE', self.root), patch.object(services.psutil, 'Process', return_value=process), \
                patch.object(services.psutil, 'wait_procs', return_value=([], [])):
            manager = services.Manager()
            with patch.object(manager, 'discover', return_value=[process]):
                manager.submit('experiments', 'stop', 123, 456)
                self.assertTrue(manager.locks['experiments'].acquire(timeout=3))
                manager.locks['experiments'].release()
        process.terminate.assert_called_once()
        process.children.assert_not_called()

    def test_stop_all_does_not_stop_another_dashboard_type(self):
        rows = [dict(kind='tensorboard', pid=123, created=1), dict(kind='optuna', pid=456, created=2)]
        process = MagicMock()
        process.create_time.return_value = 1
        process.children.return_value = []
        with patch.object(self.manager, 'discover', return_value=rows), \
                patch('psutil.Process', return_value=process) as lookup, \
                patch('psutil.wait_procs', return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                self.manager.act({'action': 'stop_all', 'kind': 'tensorboard',
                                  'targets': [{'pid': 123, 'created': 1}, {'pid': 456, 'created': 2}]})
            lookup.assert_called_once_with(123)
            process.terminate.assert_called_once()

    def test_stale_identity_never_stops_process(self):
        with patch.object(self.manager, 'discover', return_value=[dict(kind='optuna', pid=123, created=200)]), patch('psutil.Process') as process:
            with self.assertRaisesRegex(ValueError, 'identity'):
                self.manager.act({'kind': 'optuna', 'action': 'stop', 'pid': 123, 'created': 100})
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
