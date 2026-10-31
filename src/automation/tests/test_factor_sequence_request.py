"""One-shot validation handoff without production requests, training or mail."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from src.automation.scheduling import factor_sequence as queue


class FactorSequenceRequestTests(unittest.TestCase):
    def test_submit_and_observe_attempt_once(self):
        with TemporaryDirectory() as tmp, patch.object(queue, 'requests_dir', return_value=Path(tmp)):
            path = queue.submit('test_01')
            self.assertEqual(path, queue.submit('test_01'))
            store = Mock(); store.owned.return_value = []
            request = queue.observe(store)
            self.assertEqual(request['task_kind'], 'factor_sequence')
            store.owned.return_value = [{**request, 'phase': 'failed'}]
            self.assertIsNone(queue.observe(store))
            store.owned.return_value = [{**request, 'phase': 'deferred'}]
            self.assertEqual(queue.observe(store)['request_id'], 'test_01')
            path.write_text(json.dumps({'id': 'other', 'schema_version': 1}))
            with self.assertRaises(ValueError):
                queue.submit('test_01')

    def test_busy_slot_does_not_start(self):
        with patch('src.proj.util.script.FitLock.is_held', return_value=True), \
             patch('src.proj.util.filesys.dynamic_import.dynamic_modules') as modules:
            self.assertEqual(queue.execute({'request_id': 'test'}, Mock()), 75)
            modules.assert_not_called()

    def test_failed_run_releases_slot_and_records_delivered_mail(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); result = root/'factor_sequence/test'; result.mkdir(parents=True)
            (result/'mail_status.json').write_text('{"status":"sent"}')
            (result/'suite.json').write_text('{"status":"failed"}')
            module = SimpleNamespace(main=Mock(side_effect=RuntimeError('test failure')))
            argv = sys.argv
            with patch('src.proj.PATH.result', root), \
                 patch('src.proj.util.script.FitLock.is_held', return_value=False), \
                 patch('src.proj.util.script.FitLock.guard') as guard, \
                 patch('src.proj.util.filesys.dynamic_import.dynamic_modules', return_value=[module]), \
                 patch('src.automation.scheduling.runtime.record_script_email') as receipt:
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    queue.execute({'request_id': 'test', 'id': 'managed_test'}, Mock())
                guard.return_value.__exit__.assert_called_once()
                module.main.assert_called_once_with(run_id='test')
                receipt.assert_called_once_with(success=False)
            self.assertIs(sys.argv, argv)


if __name__ == '__main__':
    unittest.main()
