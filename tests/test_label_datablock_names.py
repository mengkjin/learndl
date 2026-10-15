"""Canonical label dump names and value-preserving filename migration."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from src.data import DataBlock
from src.data.util.classes import data_block

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/.autofix/rename_label_datablocks.py'
spec = importlib.util.spec_from_file_location('rename_label_datablocks', SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


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

    def test_migrate_all_formats_preserves_bytes_inode_and_unrelated_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'blocks'
            originals = []
            for frame in ('fit', 'predict', 'raw'):
                folder = root / frame
                folder.mkdir(parents=True)
                for suffix in migration.SUFFIXES:
                    path = folder / f'X_y_open{suffix}'
                    if suffix == '.mmap':
                        path.mkdir()
                        (path / 'values.bin').write_bytes(b'tensor-content')
                    else:
                        path.write_bytes(b'serialized-content')
                    if frame != 'raw':
                        originals.append((path, path.stat().st_ino))
                (folder / 'X_day.pt').write_bytes(b'day')
                (folder / 'Y.pt').write_bytes(b'default-labels')
            reports = Path(directory) / 'reports'
            self.assertEqual(len(migration.migrate(root, reports, dry_run=True)), 6)
            self.assertFalse(reports.exists())
            moves = migration.migrate(root, reports)
            self.assertEqual(len(moves), 6)
            for source, inode in originals:
                target = source.with_name(source.name.replace('X_y_', 'Y_', 1))
                self.assertFalse(source.exists())
                self.assertEqual(target.stat().st_ino, inode)
                payload = target / 'values.bin' if target.is_dir() else target
                self.assertEqual(payload.read_bytes(), b'tensor-content' if target.is_dir() else b'serialized-content')
            self.assertTrue((root / 'raw/X_y_open.mmap').exists())
            self.assertEqual((root / 'fit/X_day.pt').read_bytes(), b'day')
            self.assertEqual((root / 'predict/Y.pt').read_bytes(), b'default-labels')
            report = json.loads(next(reports.glob('*.json')).read_text())
            self.assertEqual(report['status'], 'success')
            self.assertEqual(migration.migrate(root, reports), [])

    def test_conflict_in_other_format_aborts_before_any_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fit').mkdir()
            (root / 'predict').mkdir()
            (root / 'fit/X_y_lag0.pt').write_bytes(b'first')
            (root / 'predict/X_y_vwap.pt').write_bytes(b'old')
            (root / 'predict/Y_vwap.feather').write_bytes(b'new')
            with self.assertRaises(FileExistsError):
                migration.migrate(root, root / 'reports')
            self.assertEqual((root / 'fit/X_y_lag0.pt').read_bytes(), b'first')
            self.assertEqual((root / 'predict/Y_vwap.feather').read_bytes(), b'new')
            self.assertFalse((root / 'reports').exists())

    def test_symlinks_are_not_renamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fit').mkdir()
            (root / 'fit/X_y_open.pt').symlink_to(root / 'missing')
            with self.assertRaises(ValueError):
                migration.plan_renames(root)


if __name__ == '__main__':
    main()
