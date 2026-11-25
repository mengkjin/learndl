"""Exercise the repaired loader and the 0.7.11 rebuild with real DataBlock dumps."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.automation.version_patch import load_patch_support
from src.data.loader import FactorLoader
from src.data.util import DataBlock
from src.proj import CALENDAR, DB, Dates, PATH


migration = load_patch_support('0.7.11', 'factor_datablocks')


class FactorDataBlocksPatchTest(TestCase):
    key = 'momentum'
    factor = 'umr_new_1m'
    source_dates = [20260413, 20260414, 20260416]

    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ctx = SimpleNamespace(root=root, backup_dir=root / 'backup')
        self.ctx.backup_dir.mkdir()
        self.enterContext(patch.object(PATH, 'datablock', root / 'DataBlock'))
        self.enterContext(patch.object(CALENDAR, 'updated', return_value=20260415))
        self.enterContext(patch.object(CALENDAR, 'update_to', return_value=20260415))
        self.enterContext(patch.object(DB, 'dates', side_effect=self.stored_dates))
        self.enterContext(patch.object(DB, 'loads', side_effect=self.load_factors))
        self.enterContext(patch.object(DataBlock, 'mask_values', lambda block, **_: block))
        self.enterContext(patch.object(migration.MACHINE, 'cuda_server', False))
        self.enterContext(patch.object(migration.MACHINE, 'updatable', False))

    def stored_dates(self, src, key):
        return Dates(self.source_dates if key == self.factor else [])

    def load_factors(self, src, key, dates, **kwargs):
        if key != self.factor:
            return pd.DataFrame()
        return pd.DataFrame(
            [(secid, date, float(secid)) for date in self.source_dates if date in dates
             for secid in (1, 2, 3)], columns=['secid', 'date', key])

    def seed_dump(self, frame, key=None):
        key = key or self.key
        path = DataBlock.path_preprocess(key, frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        block = DataBlock(torch.full((1, 2, 1, 1), 7.), [1], [20260409, 20260410], [self.factor])
        block.set_flags(category='preprocess', preprocess_key=key, frame=frame).save_dump()
        return path

    def test_rebuild_validates_source_dates_and_preserves_fit_history(self):
        fit_path = self.seed_dump('fit')
        predict_path = self.seed_dump('predict')
        unrelated = self.seed_dump('fit', 'style')
        fit_original = (fit_path / 'values/data.bin').read_bytes()
        unrelated_id = unrelated.stat().st_ino

        migration.apply(self.ctx)
        self.assertTrue(migration.verify(self.ctx))
        fit = DataBlock.load_preprocess(self.key, 'fit')
        predict = DataBlock.load_preprocess(self.key, 'predict')
        np.testing.assert_array_equal(fit.date, [20260409, 20260410, 20260413, 20260414])
        np.testing.assert_array_equal(predict.date, [20260413, 20260414])
        torch.testing.assert_close(fit.loc(secid=[1], date=[20260409, 20260410]), torch.full((1, 2, 1, 1), 7.))
        self.assertEqual((self.ctx.backup_dir / 'fit' / fit_path.name / 'values/data.bin').read_bytes(), fit_original)
        self.assertTrue((self.ctx.backup_dir / 'predict' / predict_path.name).is_dir())
        self.assertEqual(unrelated.stat().st_ino, unrelated_id)

        migration.apply(self.ctx)
        self.assertTrue(migration.verify(self.ctx))
        self.assertEqual((self.ctx.backup_dir / 'fit' / fit_path.name / 'values/data.bin').read_bytes(), fit_original)
        self.seed_dump('fit')
        with self.assertRaisesRegex(ValueError, '20260414'):
            migration.verify(self.ctx)

    def test_latest_source_without_usable_values_does_not_replace_dump(self):
        path = self.seed_dump('fit')
        inode = path.stat().st_ino
        self.enterContext(patch.object(DataBlock, 'mask_values', return_value=DataBlock()))
        with self.assertRaisesRegex(ValueError, 'latest stored date 20260414'):
            migration.apply(self.ctx)
        self.assertEqual(path.stat().st_ino, inode)

    def test_named_loader_keeps_factor_filters_separate_from_logging(self):
        loader = FactorLoader([self.factor], category1=self.key, notice_empty=False, indent=2, vb_level=4)
        block = loader.load(20260413, 20260414)
        self.assertEqual(list(block.feature), [self.factor])
        self.assertEqual(block.last_meaningful_date, 20260414)
