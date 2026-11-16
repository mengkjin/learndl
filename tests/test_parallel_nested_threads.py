import threading
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from src.proj.util.functional.parallel import parallel, serial_execution


class NestedThreadTest(unittest.TestCase):
    def test_normal_nested_calls_remain_parallel(self):
        def outer():
            owner = threading.get_ident()
            values = parallel([lambda: threading.get_ident() for _ in range(5)], max_workers=4)
            self.assertNotIn(owner, values.values())
            return len(values)
        with patch('src.proj.util.functional.parallel.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool:
            results = parallel([outer, outer, outer], max_workers=2)
            self.assertEqual(list(results.values()), [5, 5, 5])
            self.assertEqual(pool.call_count, 4)
            parallel([lambda: 1, lambda: 2], max_workers=2)
            self.assertEqual(pool.call_count, 5)

    def test_nested_exception_propagates(self):
        def fail():
            raise ValueError('nested failure')
        def outer():
            return parallel([fail, fail], max_workers=2)
        with self.assertRaisesRegex(ValueError, 'nested failure'):
            parallel([outer, outer], max_workers=2)

    def test_serial_context_covers_nested_threads_and_processes_and_restores(self):
        owner = threading.get_ident()
        def outer():
            return parallel([threading.get_ident, threading.get_ident], method='process')
        with patch('src.proj.util.functional.parallel.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool:
            with patch('src.proj.util.functional.parallel.ProcessPoolExecutor') as processes:
                with self.assertRaisesRegex(ValueError, 'reset'):
                    with serial_execution(), serial_execution():
                        results = parallel([outer, outer])
                        self.assertEqual(results, {0: {0: owner, 1: owner}, 1: {0: owner, 1: owner}})
                        raise ValueError('reset')
                processes.assert_not_called()
                pool.assert_not_called()
            parallel([lambda: 1, lambda: 2])
            self.assertEqual(pool.call_count, 1)
