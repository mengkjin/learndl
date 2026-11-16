import threading
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from src.proj.util.functional.parallel import parallel


class NestedThreadTest(unittest.TestCase):
    def test_nested_calls_reuse_outer_worker_and_return_all_results(self):
        def outer():
            owner = threading.get_ident()
            values = parallel([lambda: threading.get_ident() for _ in range(5)], max_workers=4)
            self.assertEqual(set(values.values()), {owner})
            return len(values)
        with patch('src.proj.util.functional.parallel.ThreadPoolExecutor', wraps=ThreadPoolExecutor) as pool:
            results = parallel([outer, outer, outer], max_workers=2)
            self.assertEqual(list(results.values()), [5, 5, 5])
            self.assertEqual(pool.call_count, 1)
            parallel([lambda: 1, lambda: 2], max_workers=2)
            self.assertEqual(pool.call_count, 2)

    def test_nested_exception_propagates(self):
        def fail():
            raise ValueError('nested failure')
        def outer():
            return parallel([fail, fail], max_workers=2)
        with self.assertRaisesRegex(ValueError, 'nested failure'):
            parallel([outer, outer], max_workers=2)
