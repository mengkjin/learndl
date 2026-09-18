import importlib.util
import unittest
from pathlib import Path

import polars as pl

spec = importlib.util.spec_from_file_location('audit_min_chars', Path(__file__).resolve().parents[1] / 'scripts/0_check/audit_min_chars.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditMinCharsTest(unittest.TestCase):
    def test_raw_mapping_distinction(self):
        df = pl.DataFrame({'secid': [900001, 2], 'date': [20220101, 20220101]})
        def mapper(x):
            return x.with_columns(pl.col('secid').replace({900001: 1}))
        report, raw, mapped, changes = audit.inspect_keys(df, 20220101, {1, 2}, mapper)
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['raw_unknown'], 1)
        self.assertEqual(raw, {900001, 2})
        self.assertEqual(mapped, {1, 2})
        self.assertEqual(changes, [(900001, 1)])

    def test_bad_daily_file(self):
        df = pl.DataFrame({'secid': [1, 1, 99], 'date': [20220101, 20220101, 20220102]})
        report, _, _, _ = audit.inspect_keys(df, 20220101, {1}, lambda x: x, 2)
        self.assertEqual(set(report['issues']), {'duplicate_keys', 'date_mismatch', 'unknown_mapped_secids', 'row_count_above_threshold'})

    def test_mapping_collision_and_failure(self):
        df = pl.DataFrame({'secid': [1, 2], 'date': [20220101, 20220101]})
        report, _, _, _ = audit.inspect_keys(df, 20220101, {1, 2}, lambda x: x.with_columns(pl.lit(1).alias('secid')))
        self.assertIn('mapping_key_collision', report['issues'])
        def broken(x):
            raise ValueError('broken mapper')
        report, raw, _, _ = audit.inspect_keys(df, 20220101, {1, 2}, broken)
        self.assertEqual(raw, {1, 2})
        self.assertIn('broken mapper', report['mapping_error'])

    def test_invalid_keys_not_coerced(self):
        df = pl.DataFrame({'secid': ['1', None], 'date': [20220101, 20220101]})
        report, _, _, _ = audit.inspect_keys(df, 20220101, {1}, lambda x: x)
        self.assertIn('invalid_secid', report['issues'])


if __name__ == '__main__':
    unittest.main()
