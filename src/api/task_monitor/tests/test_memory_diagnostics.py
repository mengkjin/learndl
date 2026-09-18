from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import polars as pl
import torch

from src.data.preprocess.mem_trace import describe
from src.data.util import DataBlock


class MemoryDiagnosticsTest(unittest.TestCase):
    def test_expanded_tensor_reports_storage_separately(self):
        tensor = torch.arange(10).reshape(10, 1).expand(10, 20)
        result = describe(tensor)
        self.assertIn('logical=', result)
        self.assertIn('storage=80B', result)

    def test_probes_preserve_polars_merge_fill_and_mask_results(self):
        frame = pl.DataFrame({'secid': [1, 2], 'date': [20200101, 20200102],
                              'feature': pl.Series([1, 2], dtype=pl.Float32)})
        expected = DataBlock.from_polars(frame).fillna(0)
        output = io.StringIO()
        with patch.dict('os.environ', {'LEARNDL_MEMORY_TRACE': '1'}), redirect_stdout(output):
            actual = DataBlock.from_polars(frame).fillna(0)
            merged = DataBlock.merge([actual, actual.copy()])
            merged.mask_values({})
        torch.testing.assert_close(expected.values, merged.values)
        self.assertIn('before-join', output.getvalue())
        self.assertIn('after-copy', output.getvalue())


if __name__ == '__main__':
    unittest.main()
