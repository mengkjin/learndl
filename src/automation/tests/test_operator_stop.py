import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from src.automation.scheduling.runtime import RunStore, timeout_message
from src.automation.scheduling import operator_stop


class OperatorStopTests(unittest.TestCase):
    def test_stop_persists_before_signal_and_survives_worker_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=RunStore(Path(tmp)/'runs.sqlite')
            run=dict(id='one',phase='running',owner='idle_worklist',pid=123,process_created=4,
                     runner_pid=122,uid=501,boot_id='boot',cgroup='/group')
            store.put(run)
            store.put(dict(id='retry',phase='queued',parent_run_id='one'))
            with patch.object(operator_stop,'store_for',return_value=store), \
                 patch.object(operator_stop,'members',return_value=[Mock()]), \
                 patch.object(operator_stop,'signal_members') as signal:
                identity=operator_stop.prepare('managed:scheduler:one',123,4)
                def check(*args):
                    self.assertTrue(store.get('one')['suppress_auto_recovery'])
                    self.assertEqual(store.get('retry')['phase'],'cancelled')
                signal.side_effect=check
                operator_stop.stop('managed:scheduler:one',identity)
                signal.assert_called_once()
                store.put(run)  # Late runner write must not remove the stop intent.
                self.assertTrue(store.get('one')['suppress_auto_recovery'])
                self.assertIn('Monitor',timeout_message(store.get('one')))
                run['pid']=987;store.put(run)
                with self.assertRaises(ValueError):
                    operator_stop.stop('managed:scheduler:one',identity)
                self.assertEqual(signal.call_count,1)

    def test_no_creation_of_missing_ledger(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(operator_stop,'runtime_dir',return_value=Path(tmp)):
            with self.assertRaises(ValueError):
                operator_stop.store_for('scheduler')
            self.assertFalse((Path(tmp)/'runs.sqlite').exists())
