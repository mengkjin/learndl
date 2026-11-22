"""Behavior contracts for standalone utilities shared across project packages."""
from __future__ import annotations

import json
import tempfile
import unittest
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.shared import cached_properties, duration, filtered, singleton
from src.shared.dynamic_import import dynamic_members, dynamic_modules
from src.shared.file_ops import atomic_bytes, atomic_json
from src.shared.resource.memory import allocation_budget, available_bytes, fmt_bytes, hwm_bytes, rss_bytes


class FilePublicationTest(unittest.TestCase):
    def test_preserves_utf8_json_and_binary_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'nested', 'output')
            value = {'报告': ['é', {'count': 7}]}
            atomic_json(path, value)
            self.assertEqual(path.read_bytes(), json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))
            atomic_bytes(str(path), b'\xff\x00\r\n')
            self.assertEqual(path.read_bytes(), b'\xff\x00\r\n')

    def test_failure_before_replace_preserves_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'output')
            path.write_bytes(b'previous')
            for target in ('os.fsync', 'os.replace'):
                with self.subTest(target=target), patch(f'src.shared.file_ops.{target}', side_effect=OSError('disk error')):
                    with self.assertRaisesRegex(OSError, 'disk error'):
                        atomic_bytes(path, b'new')
                    self.assertEqual(path.read_bytes(), b'previous')
                    self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises(TypeError):
                atomic_json(path, {'valid': 1, 'invalid': object()})
            self.assertEqual(path.read_bytes(), b'previous')
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_concurrent_writers_publish_complete_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'output')
            versions = [{'version': i, 'content': str(i) * 10000} for i in range(12)]
            atomic_json(path, versions[0])

            def publish(value):
                atomic_json(path, value)
                return json.loads(path.read_bytes())

            with ThreadPoolExecutor(max_workers=6) as workers:
                for observed in workers.map(publish, versions):
                    self.assertIn(observed, versions)
            self.assertIn(json.loads(path.read_bytes()), versions)
            self.assertEqual(list(path.parent.iterdir()), [path])


class ResourceTest(unittest.TestCase):
    def test_process_memory_and_platform_fallback(self):
        status = 'Name:   python\nVmHWM:  1200 kB\nVmRSS:  1024 kB\n'
        with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'read_text', return_value=status):
            self.assertEqual(fmt_bytes(rss_bytes()), '1.0M')
            self.assertEqual(hwm_bytes(), 1200 * 1024)
        with patch.object(Path, 'exists', return_value=False), patch(
            'src.shared.resource.memory.subprocess.check_output', return_value=' 512\n'
        ):
            self.assertEqual(rss_bytes(), 512 * 1024)
            self.assertIsNone(hwm_bytes())

    def test_available_memory_fallback_and_explicit_budget(self):
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=1000)), patch.object(
            Path, 'read_text', side_effect=FileNotFoundError
        ):
            self.assertEqual(available_bytes(), 1000)
        self.assertEqual(allocation_budget(1000, fraction=.5, limit=None), 500)
        self.assertEqual(allocation_budget(1000, fraction=.8, limit=300), 300)
        self.assertEqual(allocation_budget(1000, fraction=.8, limit=0), 0)


class PythonUtilityTest(unittest.TestCase):
    def test_timing_format_and_existing_exports(self):
        from src.proj import Base
        from src.proj import core

        self.assertIs(Base.Elapsed, duration.Elapsed)
        self.assertIs(Base.Since, duration.Since)
        self.assertIs(core.SingletonMeta, singleton.SingletonMeta)
        self.assertIs(Base.Singleton, singleton.SingletonMeta)
        self.assertIs(Base.SingletonABC, singleton.SingletonABCMeta)
        self.assertIs(Base.NoInstance, singleton.NoInstanceMeta)
        self.assertIs(Base.CacheProps, cached_properties.CachedProperties)
        self.assertIs(Base.FilteredIterable, filtered.FilteredIterable)
        self.assertEqual(str(duration.Elapsed(.25)), '<1 Sec')
        self.assertEqual(str(duration.Elapsed(.25, high_precision=True)), '0.2500 Sec')
        self.assertEqual(str(duration.Elapsed(timedelta(days=1, seconds=61))), '1 Day 1 Min 1 Sec')
        self.assertEqual(duration.Since().seconds, 0)

    def test_singleton_is_shared_by_concurrent_callers(self):
        class Service(metaclass=singleton.SingletonMeta):
            created = 0

            def __init__(self):
                type(self).created += 1

        class AbstractService(metaclass=singleton.SingletonABCMeta):
            @abstractmethod
            def run(self):
                pass

        with ThreadPoolExecutor(max_workers=8) as workers:
            instances = list(workers.map(lambda _: Service(), range(40)))
        self.assertEqual(Service.created, 1)
        self.assertTrue(all(instance is instances[0] for instance in instances))
        with self.assertRaises(TypeError):
            AbstractService()  # type: ignore[abstract]

    def test_grouped_cache_and_filtering_contract(self):
        cache = cached_properties.CachedProperties().cached_properties
        marker = object()
        self.assertIs(cache.query('items', 'key', lambda: marker), marker)
        with patch.object(cache, 'set') as set_value:
            self.assertIs(cache.query('items', 'key', object), marker)
            set_value.assert_not_called()
        cache.set('key', 3)
        cache.clear('items')
        self.assertFalse(cache.has('items', 'key'))
        self.assertEqual(cache.get('key'), 3)
        cache.clear_all()
        self.assertFalse(cache.has('key'))
        selected = filtered.FilteredIterable([1, 2, 3], [False, True])
        self.assertEqual(list(selected), [2])
        self.assertEqual(list(selected), [])
        self.assertEqual(list(filtered.FilteredIterable([1, 2, 3], lambda value: value > 1)), [2, 3])

    def test_dynamic_modules_are_loaded_in_sorted_order_and_filter_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            later = root / 'z.py'
            earlier = root / 'a.py'
            later.write_text('def last(): pass\n')
            earlier.write_text('from pathlib import Path\nclass Child(dict): pass\n')
            self.assertEqual([module.__name__ for module in dynamic_modules(str(root))], [str(earlier), str(later)])
            self.assertEqual([name for name, _ in dynamic_members(earlier)], ['Child'])
            self.assertEqual([name for name, _ in dynamic_members(root, subclass_of=dict)], ['Child'])
            later.write_text('raise RuntimeError("script failed")\n')
            with self.assertRaisesRegex(RuntimeError, 'script failed'):
                list(dynamic_modules(later))
