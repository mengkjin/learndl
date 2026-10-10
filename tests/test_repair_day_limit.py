"""Day-limit repairs must be reversible and must not invent conflict resolution."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/2_data/7_repair_day_limit.py'
spec = importlib.util.spec_from_file_location('repair_day_limit_script', SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class DayLimitRepairTest(unittest.TestCase):
    def test_backup_dry_run_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, backup = root / 'limit.feather', root / 'backup/limit.feather'
            frame = pd.DataFrame({'secid': [1, 1, 2], 'up_limit': [11., 11., 22.], 'down_limit': [9., 9., 18.]})
            repair.dfIOHandler.save_df(frame, source)
            before = source.read_bytes()
            self.assertEqual(repair.repair_file(source, backup, 20240101, True)['removed'], 1)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(backup.exists())
            self.assertEqual(repair.repair_file(source, backup, 20240101, False)['status'], 'repaired')
            self.assertEqual(backup.read_bytes(), before)
            self.assertEqual(len(repair.dfIOHandler.load_pandas(source)), 2)
            self.assertEqual(repair.repair_file(source, backup, 20240101, False)['status'], 'unchanged')

    def test_conflict_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, backup = root / 'limit.feather', root / 'backup/limit.feather'
            frame = pd.DataFrame({'secid': [1, 1], 'up_limit': [11., 12.], 'down_limit': [9., 9.]})
            repair.dfIOHandler.save_df(frame, source)
            before = source.read_bytes()
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                repair.repair_file(source, backup, 20240101, False)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(backup.exists())

    def test_date_mismatch_and_null_id_rejected(self):
        frame = pd.DataFrame({'secid': [1], 'up_limit': [11.], 'down_limit': [9.], 'date': [20240102]})
        with self.assertRaisesRegex(ValueError, 'date column'):
            repair.clean_day_limit(frame, 20240101)
        frame.loc[0, 'secid'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'null secid'):
            repair.clean_day_limit(frame, 20240102)


if __name__ == '__main__':
    unittest.main()
