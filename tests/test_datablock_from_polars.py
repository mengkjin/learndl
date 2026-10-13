"""Duplicate recovery is deferred until dense DataBlock reshaping fails."""
import unittest
from unittest.mock import patch

import polars as pl
import torch

from src.data.util import DataBlock
from src.proj import Logger


class DataBlockFromPolarsTest(unittest.TestCase):
    def test_normal_conversion_does_not_check_duplicates(self):
        raw = pl.DataFrame({'secid': [10, 20], 'date': [20240102, 20240103], 'score': [2., 6.]})
        with patch.object(pl.DataFrame, 'is_duplicated', side_effect=AssertionError('Unexpected check')), \
             patch.object(pl.DataFrame, 'unique', side_effect=AssertionError('Unexpected dedup')), \
             patch.object(Logger, 'warning') as warning:
            block = DataBlock.from_polars(raw)
        warning.assert_not_called()
        torch.testing.assert_close(block.values[:, :, 0, 0],
                                   torch.tensor([[2., float('nan')], [float('nan'), 6.]], dtype=block.values.dtype),
                                   equal_nan=True)

    def test_identical_duplicates_warn_all_dates_and_recover(self):
        raw = pl.DataFrame({'secid': [10, 20], 'date': [20240102, 20240103], 'score': [2., float('nan')]})
        expected = DataBlock.from_polars(raw)
        with patch.object(Logger, 'warning') as warning:
            block = DataBlock.from_polars(pl.concat([raw, raw]))
        warning.assert_called_once()
        self.assertIn('[20240102, 20240103]', warning.call_args.args[0])
        torch.testing.assert_close(block.values, expected.values, equal_nan=True)

    def test_conflicts_are_not_silently_dropped(self):
        raw = pl.DataFrame({'secid': [10, 10, 10], 'date': [20240102] * 3, 'score': [2., 2., 3.]})
        with patch.object(Logger, 'warning') as warning, \
             self.assertRaisesRegex(ValueError, 'conflicting values.*20240102'):
            DataBlock.from_polars(raw)
        warning.assert_called_once()

    def test_intraday_rows_remain_distinct(self):
        # The same stock/date at different minutes is valid, including on retry.
        for mark in ['inday', 'minute']:
            with self.subTest(mark=mark):
                raw = pl.DataFrame({'secid': [10, 10], 'date': [20240102] * 2,
                                    mark: [1, 2], 'score': [2., 3.]})
                expected = DataBlock.from_polars(raw)
                with patch.object(Logger, 'warning'):
                    block = DataBlock.from_polars(pl.concat([raw, raw.head(1)]))
                torch.testing.assert_close(block.values, expected.values)

    def test_unrelated_reshape_failure_is_reraised(self):
        raw = pl.DataFrame({'secid': [10], 'date': [20240102], 'score': [2.]})
        with patch.object(torch.Tensor, 'reshape', side_effect=RuntimeError('unrelated reshape failure')), \
             patch.object(Logger, 'warning') as warning, \
             self.assertRaisesRegex(RuntimeError, 'unrelated reshape failure'):
            DataBlock.from_polars(raw)
        warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
