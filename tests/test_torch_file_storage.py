"""Production batch-cache contract; all disk operations stay in temporary roots."""
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import torch

from src.res.model.util.storage.torch_file import TorchFileStorage, StoredTorchFileLoader


class TemporaryStorage(TorchFileStorage):
    def __init__(self, memory, directory):
        super().__init__(memory)
        self.directory = Path(directory)

    def real_path(self, key):
        return self.directory / f'{key}.pt'


class TorchFileStorageTests(TestCase):
    def test_overwrite_move_groups_and_delete(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                store.save(torch.tensor([1.]), 'a', 'g0')
                store.save(torch.tensor([2.]), 'b', 'g1')
                store.save(torch.tensor([3.]), 'c', 'g0')
                store.save(torch.tensor([4.]), 'a', 'g0')
                store.save(torch.tensor([5.]), 'b', 'g0')
                self.assertEqual(store.keys, ['a', 'b', 'c'])
                self.assertEqual(store.records['key'].tolist(), ['c', 'a', 'b'])
                self.assertEqual(store.records.index.tolist(), [0, 0, 0])
                store.del_group('g1')  # Overwritten b must not remain in old group.
                self.assertTrue(store.exists('b'))
                torch.testing.assert_close(store.load('b'), torch.tensor([5.]))
                store.save(torch.tensor([6.]), 'c', 'g2')
                store.del_group('g0')
                self.assertEqual(store.keys, ['c'])
                self.assertFalse(store.exists('a'))
                store.del_group('g0')
                store.del_group('missing')
                with self.assertRaises(KeyError): store.del_one('missing')
                self.assertEqual(store.keys, ['c'])
                store.del_one('c')
                store.del_all()
                self.assertTrue(store.records.empty)
                store.save(torch.tensor([7.]), 'd')
                store.del_all()
                store.del_all()
                self.assertEqual(store.keys, [])
                self.assertEqual(store._groups, {})
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_dataframe_is_independent_query_snapshot(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                expected_empty = pd.DataFrame(columns=pd.Index(['key', 'path', 'group']), dtype=str)
                pd.testing.assert_frame_equal(store.records, expected_empty)
                store.save(torch.ones(2), 'a', 'g')
                store.save(torch.zeros(2), 'b', 'other')
                path = 'a' if memory else str(Path(root) / 'a.pt')
                expected = pd.DataFrame([{'key':'a', 'path':path, 'group':'g'}], index=[0])
                pd.testing.assert_frame_equal(store.records.query("group == 'g'"), expected)
                snapshot = store.records
                snapshot.loc[:, 'group'] = 'changed'
                self.assertEqual(store.records['group'].tolist(), ['g', 'other'])
                store.save(torch.ones(2), 'a', 'new')
                self.assertEqual(snapshot['key'].tolist(), ['a', 'b'])
                self.assertEqual(store.records['key'].tolist(), ['b', 'a'])

    def test_static_loader_and_missing_load(self):
        for memory in (True, False):
            with self.subTest(memory=memory), tempfile.TemporaryDirectory() as root:
                store = TemporaryStorage(memory, root)
                for i in range(3): store.save(torch.tensor([i]), str(i), 'fit')
                loader = StoredTorchFileLoader(store, ['2', '0', '1'])
                self.assertEqual(len(loader), 3)
                self.assertEqual([int(x.item()) for x in loader], [2, 0, 1])
                self.assertEqual(loader[1].item(), 0)
                if memory:
                    with self.assertRaises(KeyError): store.load('missing')
                else:
                    self.assertIsNone(store.load('missing'))
                    store.real_path('1').unlink()
                    self.assertIsNone(store.load('1'))
                store.del_all()

    def test_mutation_does_not_materialize_dataframe(self):
        with tempfile.TemporaryDirectory() as root:
            store = TemporaryStorage(True, root)
            with patch('src.res.model.util.storage.torch_file.pd.DataFrame', side_effect=AssertionError('hot-path table')):
                for i in range(480): store.save(None, str(i), f'g{i%2}')
                store.save(None, '0', 'new')
                store.del_group('g0')
                store.del_one('0')
                store.del_all()
            self.assertTrue(store.records.empty)

    def test_failed_payload_operations_preserve_index(self):
        with tempfile.TemporaryDirectory() as root:
            store = TemporaryStorage(False, root)
            store.save(torch.ones(1), 'a', 'original')
            before = store.records
            with patch('torch.save', side_effect=OSError('write failed')):
                with self.assertRaises(OSError): store.save(torch.zeros(1), 'a', 'new')
            pd.testing.assert_frame_equal(store.records, before)
            with patch.object(Path, 'unlink', side_effect=OSError('unlink failed')):
                with self.assertRaises(OSError): store.del_one('a')
            pd.testing.assert_frame_equal(store.records, before)
            self.assertTrue(store.exists('a'))
            store.del_group('original')
            self.assertTrue(store.records.empty)
