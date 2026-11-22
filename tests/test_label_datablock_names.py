"""Canonical label dump names and value-preserving filename migration."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from src.data import DataBlock
from src.data.util.classes import data_block

class LabelDumpNamesTest(TestCase):
    def test_canonical_names_and_legacy_read_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(data_block, 'PATH', SimpleNamespace(datablock=Path(directory))):
            root = Path(directory)
            for frame in ('fit', 'predict'):
                folder = root / frame
                folder.mkdir()
                for key in ('y_open', 'y_vwap', 'y_lag0', 'y_custom'):
                    canonical = folder / f'Y_{key[2:]}.mmap'
                    self.assertEqual(DataBlock.path_preprocess(key, frame), canonical)
                    old = folder / f'X_{key}.mmap'
                    old.mkdir()
                    self.assertEqual(DataBlock.path_preprocess(key, frame), old)
                    self.assertEqual(DataBlock.path_preprocess(key, frame, find_if_not_exists=False), canonical)
                    new = folder / f'Y_{key[2:]}.pt'
                    new.touch()
                    self.assertEqual(DataBlock.path_preprocess(key, frame), new)
                self.assertEqual(DataBlock.path_preprocess('y', frame), folder / 'Y.mmap')
                self.assertEqual(DataBlock.path_preprocess('labels', frame), folder / 'Y.mmap')
                self.assertEqual(DataBlock.path_preprocess('day', frame), folder / 'X_day.mmap')

    def test_load_legacy_then_save_canonical_without_changing_legacy(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(data_block, 'PATH', SimpleNamespace(datablock=Path(directory))):
            folder = Path(directory) / 'fit'
            folder.mkdir()
            legacy = folder / 'X_y_open.pt'
            data = {'values': torch.ones(2, 1, 1, 1), 'secid': np.array([1, 2]), 'date': np.array([20240102]), 'feature': np.array(['std_10_open'])}
            torch.save(data, legacy)
            before = legacy.read_bytes()
            block = DataBlock.load_dump(category='preprocess', preprocess_key='y_open', frame='fit')
            block.values += 1
            block.save_dump()
            self.assertEqual(legacy.read_bytes(), before)
            self.assertTrue((folder / 'Y_open.pt').exists())
            loaded = DataBlock.load_dump(category='preprocess', preprocess_key='y_open', frame='fit')
            torch.testing.assert_close(loaded.values, torch.full((2, 1, 1, 1), 2.))


if __name__ == '__main__':
    main()
