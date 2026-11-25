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


migration = load_patch_support('0.7.11', 'factor_datablocks')
repair = load_patch_support('0.7.13', 'turnover_snapshot')


class FactorDataBlocksPatchTest(TestCase):
    key = 'momentum'
    factor = 'umr_new_1m'
    source_dates = [20260413, 20260414, 20260416]

    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ctx = SimpleNamespace(root=root, backup_dir=root / 'backup')
        self.ctx.backup_dir.mkdir()
        self.enterContext(patch.object(PATH, 'export', root / 'Export'))
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
    dates = [20260619, repair.DATE]

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(PATH, 'export', self.root / 'Export'))
        self.enterContext(patch.object(PATH, 'datablock', self.root / 'DataBlock'))
        self.calc = FactorCalculator.get('turn_1m')

    def write_source(self, calc, date, values, **metadata):
        path = calc.target_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'secid': [1, 2, 3], calc.factor_name: values, **metadata}).to_feather(path)

    def test_normalized_load_rejects_unknown_columns_and_text_signals(self):
        date = self.dates[-1]
        self.write_source(self.calc, date, [1., 2., 3.], level_0=['a', 'b', 'b'])
        with self.assertRaisesRegex(ValueError, 'turn_1m.*level_0'):
            self.calc.Loads([date], normalize=True)
        for text in ('2.0', 'unparsed'):
            self.write_source(self.calc, date, ['1', text, None])
            with self.assertRaisesRegex(TypeError, 'turn_1m.*numeric'):
                self.calc.Loads([date], normalize=True)
            self.assertEqual(self.calc.Loads([date], normalize=False)['turn_1m'].iloc[1], text)
        self.write_source(self.calc, date, [1., 2., 3.], turn_1m__source_key='archive')
        self.assertNotIn('turn_1m__source_key', self.calc.Loads([date], normalize=True))

    def test_only_normalization_repairs_a_legacy_factor_and_preserves_daily_files(self):
        calc = FactorCalculator.get('turn_3m')
        clean, legacy = self.dates
        self.write_source(calc, clean, [4., 2., 6.])
        self.write_source(calc, legacy, [3., None, 1.], turn_3m__source_key='archive')
        expected = calc.Loads(self.dates, normalize=True)
        path = calc.target_path(legacy)
        clean_inode = calc.target_path(clean).stat().st_ino
        original = pd.read_feather(path)
        original.assign(level_0=['some_field', None, 'some_field']).to_feather(path)
        damaged = path.read_bytes()
        self.assertIn('level_0', calc.Load(legacy))
        self.assertIn('level_0', calc.Loads(self.dates))
        self.assertEqual(path.read_bytes(), damaged)
        pd.testing.assert_frame_equal(calc.Loads(self.dates, normalize=True), expected)
        pd.testing.assert_frame_equal(pd.read_feather(path), original, check_exact=True)
        self.assertEqual(calc.target_path(clean).stat().st_ino, clean_inode)
        inode = path.stat().st_ino
        calc.Loads(self.dates, normalize=True)
        self.assertEqual(path.stat().st_ino, inode)
        backup, = path.parent.glob('*.level_0.bak')
        self.assertEqual(backup.read_bytes(), damaged)

    def test_snapshot_repair_preserves_rows_on_retry_and_rejects_unknown_layout(self):
        ctx = SimpleNamespace(backup_dir=self.root / 'backup')
        path = self.calc.target_path(repair.DATE)
        repair.apply(ctx)
        self.assertTrue(repair.verify(ctx))
        self.write_source(self.calc, repair.DATE, [3., None, 1.], level_0=['turn_fr', None, 'turn_fr'])
        original = path.read_bytes()
        repair.apply(ctx)
        self.assertTrue(repair.verify(ctx))
        repaired = path.read_bytes()
        repair.apply(ctx)
        self.assertEqual(path.read_bytes(), repaired)
        self.assertEqual((ctx.backup_dir / 'turn_1m' / path.name).read_bytes(), original)
        for labels, values, message in (
            (['turn_fr', None, 'other'], [3., None, None], 'unexpected level_0.*other'),
            (['turn_fr', None, 'turn_fr'], [3., 2., 1.], "unexpected level_0.*'secid': 2"),
            ([None] * 3, [float('nan')] * 3, 'no turn_fr labels'),
        ):
            with self.subTest(labels=labels):
                self.write_source(self.calc, repair.DATE, values, level_0=labels)
                invalid = path.read_bytes()
                with self.assertRaisesRegex(ValueError, message):
                    repair.apply(ctx)
                self.assertEqual(path.read_bytes(), invalid)

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
                legacy = calc is self.calc and date == repair.DATE
                self.write_source(calc, date, [1., 2., 3.], **({'level_0': 'turn_fr'} if legacy else {}))

        def attempt(name):
            backup_dir = self.root / name
            backup_dir.mkdir()
            return SimpleNamespace(root=self.root, backup_dir=backup_dir)

        def seed_history():
            PATH.datablock.mkdir()
            for key in keys:
                DataBlock(torch.full((1, 1, 1, 1), 7.), [1], [20260410], ['history']).set_flags(
                    category='preprocess', preprocess_key=key, frame='fit').save_dump()

        seed_history()
        first = attempt('first')
        with (patch.object(repair, 'apply'),
              patch('src.res.factor.calculator.snapshot_repair.repair_level_zero'),
              self.assertRaisesRegex(ValueError, 'level_0')):
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
