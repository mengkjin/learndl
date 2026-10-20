"""Regression coverage for interrupted archive writes and daily summaries."""
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.proj.db.io.tarfile import save_dfs_to_tar, load_dfs_from_tar, load_tar_meta
from src.res.factor.util.agency.portfolio_accountant import PortfolioAccount


class AccountArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'account.tar'
        self.old = pd.DataFrame({'value': [1.0]})

    def save(self, df, **kwargs):
        return save_dfs_to_tar({'basic': df}, self.path, **kwargs)

    def test_round_trip_and_overwrite_false(self):
        self.save(self.old, meta={'version': 1})
        self.assertFalse(self.save(pd.DataFrame({'value': [2.0]}), overwrite=False))
        pd.testing.assert_frame_equal(load_dfs_from_tar(self.path)['basic'], self.old)
        self.assertEqual(load_tar_meta(self.path), {'version': 1})
        self.save(pd.DataFrame({'value': [2.0]}), meta={'version': 2})
        self.assertEqual(load_dfs_from_tar(self.path)['basic'].iloc[0, 0], 2.0)
        self.assertEqual(load_tar_meta(self.path), {'version': 2})
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_failed_write_preserves_previous_archive(self):
        self.save(self.old)
        original = self.path.read_bytes()
        # Metadata fails after the dataframe has already been written.
        with self.assertRaises(TypeError):
            self.save(pd.DataFrame({'value': [2.0]}), meta={'bad': object()})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_failed_first_write_does_not_publish_partial_archive(self):
        with self.assertRaises(TypeError):
            self.save(self.old, meta={'bad': object()})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_readers_see_old_archive_until_replacement(self):
        self.save(self.old)
        from src.proj.db.io.tarfile import dfIOHandler
        original_save = dfIOHandler.save_df
        def save_and_read(*args, **kwargs):
            pd.testing.assert_frame_equal(load_dfs_from_tar(self.path)['basic'], self.old)
            return original_save(*args, **kwargs)
        with patch.object(dfIOHandler, 'save_df', side_effect=save_and_read):
            self.save(pd.DataFrame({'value': [2.0]}))
        self.assertEqual(load_dfs_from_tar(self.path)['basic'].iloc[0, 0], 2.0)

    def test_summary_skips_bad_archives_and_logs_paths(self):
        good = self.root / 'good.tar'
        # A valid empty account is sufficient to exercise actual archive loading.
        save_dfs_to_tar({}, good)
        self.path.touch()
        broken = self.root / 'broken.tar'
        broken.write_bytes(b'invalid tar')
        paths = {'empty': self.path, 'broken': broken, 'good': good}
        with patch.object(PortfolioAccount, 'eval_period_ret', return_value=pd.Series({'Total': 0.1})), \
             patch('src.res.factor.util.agency.portfolio_accountant.Logger.warning') as warning:
            result = PortfolioAccount.EvalPeriodRet(paths, skip_unreadable=True)
        self.assertEqual(list(result.columns), ['good'])
        self.assertEqual(warning.call_count, 2)
        self.assertIn(str(self.path), warning.call_args_list[0].args[0])
        self.assertIn(str(broken), warning.call_args_list[1].args[0])
        with self.assertRaises(tarfile.ReadError):
            PortfolioAccount.EvalPeriodRet(paths)
        with self.assertRaises(tarfile.ReadError):
            PortfolioAccount.load(self.path)

    def test_all_unreadable_accounts_return_empty_summary(self):
        self.path.touch()
        with patch('src.res.factor.util.agency.portfolio_accountant.Logger.warning'):
            result = PortfolioAccount.EvalPeriodRet({'empty': self.path}, skip_unreadable=True)
        self.assertTrue(result.empty)

    def test_unexpected_errors_are_not_suppressed(self):
        self.path.touch()
        with patch.object(PortfolioAccount, 'load', side_effect=ValueError('invalid schema')):
            with self.assertRaisesRegex(ValueError, 'invalid schema'):
                PortfolioAccount.EvalPeriodRet({'bad': self.path}, skip_unreadable=True)


if __name__ == '__main__':
    unittest.main()
