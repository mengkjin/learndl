"""Archived TensorBoard menus group model folders without reading event files."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.api.pkgs.dashboard import archived_tensorboard_index


def _write_snapshot(model_dir: Path) -> None:
    run = model_dir / 'snapshot' / 'tensorboard' / 'gru.0.20240101.Trial0-0'
    run.mkdir(parents=True)
    (run / 'events.out.tfevents.1').write_text('a', encoding='utf-8')
    (run / 'events.out.tfevents.2').write_text('b', encoding='utf-8')


class ArchivedTensorboardIndexTest(unittest.TestCase):
    def test_groups_by_module_model_and_newest_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / 'nn@gru@gru_day.20240101120000'
            newer = root / 'nn@gru@gru_day.20240601120000'
            indexed = root / 'nn@gru@gru_day@2.20240602120000'
            short = root / 'st@nn@astgnn@min_test.20240301120000'
            for folder in (older, newer, indexed, short):
                _write_snapshot(folder)
            (root / 'not-an-archive').mkdir()
            thin = root / 'nn@gru@empty_logs.20240501120000'
            (thin / 'snapshot' / 'tensorboard' / 'run').mkdir(parents=True)
            (thin / 'snapshot' / 'tensorboard' / 'run' / 'only-one').write_text('x', encoding='utf-8')

            grouped = archived_tensorboard_index(root)

        self.assertEqual(
            [path.name for path in grouped['nn@gru']['gru_day']],
            ['nn@gru@gru_day.20240601120000', 'nn@gru@gru_day.20240101120000'],
        )
        self.assertEqual(
            [path.name for path in grouped['nn@gru']['gru_day@2']],
            ['nn@gru@gru_day@2.20240602120000'],
        )
        self.assertEqual(
            [path.name for path in grouped['st@nn@astgnn']['min_test']],
            ['st@nn@astgnn@min_test.20240301120000'],
        )
        self.assertNotIn('empty_logs', grouped['nn@gru'])

    def test_missing_root_is_empty(self):
        self.assertEqual(archived_tensorboard_index(Path('/no/such/archive')), {})
