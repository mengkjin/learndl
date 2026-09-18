import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet

from src.data.util.minchars_stock import stock_rows
from src.data.preprocess.minchars_input import load_selected
from src.data.update.custom.min_chars import daily, rolling, tagged
from src.data.update.custom.min_chars import _common

spec = importlib.util.spec_from_file_location('clean_min_chars', Path(__file__).resolve().parents[1] / 'scripts/0_check/clean_min_chars.py')
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


class StockFilterTest(unittest.TestCase):
    def test_subset_and_empty_reference(self):
        frame = pl.DataFrame({'secid': [1, 99], 'date': [20220101] * 2})
        self.assertEqual(stock_rows(frame, [1, 2])['secid'].to_list(), [1])
        with self.assertRaisesRegex(ValueError, 'empty'):
            stock_rows(frame, [])
        indexed = frame.to_pandas().set_index('secid')
        self.assertEqual(stock_rows(indexed, [1, 2]).index.to_list(), [1])

    def test_datablock_read_filters_without_requested_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.feather'
            pl.DataFrame({'secid': [11, 99], 'date': [20220101] * 2, 'x': [1., 2.]}).write_ipc(path)
            with patch('src.proj.db.basic.df_handler.dfHandler.default_mapper', side_effect=lambda x: x.with_columns(pl.col('secid').replace({11: 1}))):
                result = load_selected(path, ['x'], np.array([1, 2]))
            self.assertEqual(result['secid'].to_list(), [1])
            self.assertEqual(result['x'].to_list(), [1.])

    def test_all_calculation_stages_exclude_nonstocks(self):
        bars = pd.DataFrame([{'secid': s, 'minute': m, 'open': 10., 'close': 10. + m / 1000,
                              'high': 11., 'low': 9., 'volume': 100., 'amount': 1000.}
                             for s in [1, 99] for m in range(240)])
        with patch('src.data.util.minchars_stock.historical_stock_ids', return_value=np.array([1, 2])):
            with patch.object(daily.DB, 'load', return_value=bars):
                daily_result = daily.calc_min_chars(20220104)
                panel = _common.load_ret_panel(20220104)
            self.assertEqual(daily_result['secid'].to_list(), [1])
            self.assertEqual(panel['secid'].unique().to_list(), [1])
            # Old daily files can still contain non-stocks while cleanup is pending.
            polluted_daily = pd.concat([daily_result, daily_result.assign(secid=99)])
            def load(src, key, *args, **kwargs):
                return bars if key == 'min' else polluted_daily
            with patch.object(rolling, 'trailing_aligned_dates', return_value=[20220104]), patch.object(rolling.DB, 'load', side_effect=load):
                roll_result = rolling.calc_min_chars_roll(20220104, window=1)
            self.assertEqual(roll_result['secid'].to_list(), [1])
            polluted_roll = pd.concat([roll_result, roll_result.assign(secid=99)])
            with patch.object(tagged.DB, 'load', side_effect=lambda src, key, *a, **kw: bars if key == 'min' else polluted_roll):
                tag_result = tagged.calc_min_chars_tag(20220104)
            self.assertEqual(tag_result['secid'].to_list(), [1])
            with patch.object(daily.DB, 'load', return_value=bars.query('secid == 99')):
                self.assertTrue(daily.calc_min_chars(20220104).empty)


class CleanupTest(unittest.TestCase):
    def test_backup_atomic_roundtrip_and_idempotence(self):
        for suffix, writer, reader in [('.feather', feather.write_feather, feather.read_table),
                                       ('.parquet', parquet.write_table, parquet.read_table)]:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as folder:
                path, backup = Path(folder) / ('source' + suffix), Path(folder) / 'backup'
                original = pa.table({'secid': [11, 99], 'x': pa.array([float('nan'), 2.], type=pa.float32())}).replace_schema_metadata({b'test': b'preserved'})
                writer(original, path)
                original_bytes = path.read_bytes()
                def mapper(x):
                    return x.with_columns(pl.col('secid').replace({11: 1}))
                result = clean.clean_file(path, {1, 2}, mapper)
                self.assertEqual(result['removed_rows'], 1)
                self.assertEqual(path.read_bytes(), original_bytes)
                result = clean.clean_file(path, {1, 2}, mapper, apply=True, backup=backup)
                self.assertEqual(result['status'], 'cleaned')
                self.assertEqual(backup.read_bytes(), original_bytes)
                self.assertTrue(pl.from_arrow(reader(path)).equals(pl.from_arrow(original.slice(0, 1))))
                self.assertTrue(reader(path).schema.equals(original.schema, check_metadata=True))
                self.assertEqual(clean.clean_file(path, {1, 2}, mapper, apply=True, backup=backup)['status'], 'unchanged')

    def test_failed_validation_does_not_replace_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.feather'
            feather.write_feather(pa.table({'secid': [1, 99]}), path)
            before = path.read_bytes()
            with patch.object(clean.os, 'replace', side_effect=OSError('test write failure')):
                with self.assertRaises(OSError):
                    clean.clean_file(path, {1}, lambda x: x, apply=True, backup=Path(folder) / 'backup')
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
