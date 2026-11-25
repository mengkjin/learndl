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
from src.proj.core.literals import DataBlockTimeFrame
from src.res.factor.calculator import FactorCalculator
from src.res.factor.util.classes.stock_factor import StockFactor


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


class StoredStockFactorPatchTest(TestCase):
    dates = [20260413, 20260414]

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(PATH, 'export', self.root / 'Export'))
        self.enterContext(patch.object(PATH, 'datablock', self.root / 'DataBlock'))
        self.calc = FactorCalculator.get('lncap')

    def write_source(self, calc, date, values, **metadata):
        path = calc.target_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'secid': [1, 2, 3], calc.factor_name: values, **metadata}).to_feather(path)

    def test_only_signal_is_normalized_and_invalid_values_identify_source_row(self):
        for date in self.dates:
            self.write_source(self.calc, date, [1., 2., 3.])
        expected = self.calc.Loads(self.dates, normalize=True)
        for date in self.dates:
            self.write_source(self.calc, date, ['1', '2.0', '3e0'],
                              lncap__source_key=['archive', None, 'live'], lncap__source_date=[date] * 3)
        pd.testing.assert_frame_equal(self.calc.Loads(self.dates, normalize=True), expected)
        raw = self.calc.Loads(self.dates, normalize=False)
        self.assertIn('lncap__source_key', raw)
        self.assertIn('lncap__source_date', raw)
        self.assertEqual(raw['lncap'].iloc[0], '1')
        self.write_source(self.calc, self.dates[-1], ['1', 'unparsed', None])
        with self.assertRaisesRegex(ValueError, "lncap: invalid factor value 'unparsed' on 20260414 for secid 2"):
            self.calc.Loads(self.dates, normalize=True)

    def test_retry_after_fifteen_fit_blocks_matches_uninterrupted_rebuild(self):
        self.enterContext(patch.object(CALENDAR, 'updated', return_value=self.dates[-1]))
        self.enterContext(patch.object(CALENDAR, 'update_to', return_value=self.dates[-1]))
        self.enterContext(patch.object(DataBlock, 'mask_values', lambda block, **_: block))
        self.enterContext(patch.object(migration.MACHINE, 'cuda_server', True))
        self.enterContext(patch.object(migration.MACHINE, 'updatable', True))
        keys = []
        for processor_name in migration.factors.__all__:
            processor = getattr(migration.factors, processor_name)('fit')
            calc = next(FactorCalculator.iter(category1=processor.category1, category0=processor.category0))
            keys.append(processor.key)
            if processor.key == 'liquidity':
                calc = self.calc
            for date in self.dates:
                if calc.factor_name == self.calc.factor_name:
                    self.write_source(calc, date, ['1', '2', '3'], lncap__source_key='archive')
                else:
                    self.write_source(calc, date, [1., 2., 3.])

        def attempt(name):
            backup_dir = self.root / name
            backup_dir.mkdir()
            return SimpleNamespace(root=self.root, backup_dir=backup_dir)

        # Exercise the failing release's whole-frame normalization on the original Feather files.
        def legacy_load(dates, **options):
            return StockFactor.normalize_df(DB.loads(self.calc.db_src, self.calc.db_key, dates),
                                            fill_method=options['fill_method'])

        def seed_history():
            PATH.datablock.mkdir()
            for key in keys:
                DataBlock(torch.full((1, 1, 1, 1), 7.), [1], [20260410], ['history']).set_flags(
                    category='preprocess', preprocess_key=key, frame='fit').save_dump()

        seed_history()
        first = attempt('first')
        with patch.object(type(self.calc), 'Loads', side_effect=legacy_load), self.assertRaisesRegex(TypeError, 'str'):
            migration.apply(first)
        saved = {path.name for path in (first.backup_dir / 'fit').iterdir()}
        self.assertEqual(len(saved), 15)
        self.assertEqual(saved, {DataBlock.path_preprocess(key, 'fit').name for key in keys[:keys.index('liquidity')]})
        self.assertFalse((first.backup_dir / migration.REPORT).exists())

        retry = attempt('retry')
        migration.apply(retry)
        self.assertTrue(migration.verify(retry))
        with patch.object(PATH, 'datablock', self.root / 'clean-DataBlock'):
            seed_history()
            clean = attempt('clean')
            migration.apply(clean)
            self.assertTrue(migration.verify(clean))
            frames: tuple[DataBlockTimeFrame, ...] = ('fit', 'predict')
            expected: dict[tuple[str, DataBlockTimeFrame], DataBlock] = {
                (key, frame): DataBlock.load_preprocess(key, frame) for key in keys for frame in frames
            }
        self.assertEqual((retry.backup_dir / migration.REPORT).read_text(), (clean.backup_dir / migration.REPORT).read_text())
        for (key, frame), block in expected.items():
            with self.subTest(key=key, frame=frame):
                actual = DataBlock.load_preprocess(key, frame)
                for axis in ('date', 'secid', 'feature'):
                    np.testing.assert_array_equal(getattr(actual, axis), getattr(block, axis))
                torch.testing.assert_close(actual.values, block.values, equal_nan=True)
        for filename in saved:
            self.assertEqual((first.backup_dir / 'fit' / filename / 'values/data.bin').read_bytes(),
                             (clean.backup_dir / 'fit' / filename / 'values/data.bin').read_bytes())
