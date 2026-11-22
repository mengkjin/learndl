from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import polars as pl

from src.data.preprocess.minchars_input import load_selected, validate_keys
from src.data.util.memory_guard import check_allocation
from src.shared.resource.memory import available_bytes


class MinCharsInputTest(unittest.TestCase):
    def test_unknown_universe_reports_sample_without_densifying(self):
        frame = pl.DataFrame({'secid': [1, 2, 123456], 'date': [20220104] * 3, 'x': [1., 2., 3.]})
        with self.assertRaisesRegex(ValueError, '123456.*Stopped before densification'):
            validate_keys(frame, np.array([1, 2]), '20220104.feather')

    def test_duplicate_and_invalid_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_keys(pl.DataFrame({'secid': [1, 1], 'date': [20220104] * 2}), np.array([1]), 'test')
        with self.assertRaisesRegex(ValueError, 'integer'):
            validate_keys(pl.DataFrame({'secid': [1.5], 'date': [20220104]}), np.array([1]), 'test')

    def test_selected_read_preserves_mapping_and_requested_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'daily.feather'
            pl.DataFrame({'secid': [1, 2, 123456], 'date': [20220104] * 3,
                          'x': [1., 2., 3.], 'unused': ['a', 'b', 'c']}).write_ipc(path)
            with patch('src.proj.db.basic.df_handler.dfHandler.default_mapper', side_effect=lambda df: df) as mapper:
                data = load_selected(path, ['x'], np.array([1, 2]), np.array([1, 2]))
                mapper.assert_called_once()
            self.assertEqual(data.columns, ['secid', 'date', 'x'])
            self.assertEqual(data['secid'].to_list(), [1, 2])

    def test_historical_universe_does_not_require_current_listing(self):
        frame = pl.DataFrame({'secid': [1, 2], 'date': [20100104, 20100104]})
        # The caller supplies the full historical reference, not today's pool.
        validate_keys(frame, np.array([1, 2, 3]), 'test')


class MemoryGuardTest(unittest.TestCase):
    def test_large_union_fails_before_torch_allocation(self):
        with patch('src.data.util.memory_guard.available_bytes', return_value=100 * 1024**3):
            with self.assertRaisesRegex(MemoryError, '47312.*refusing'):
                check_allocation(85 * 1024**3, 'minc union=(47312,3156,1,152)')
            check_allocation(4 * 1024**3, 'small merge')
            check_allocation(8 * 1024**3, 'within processor cap', limit=8 * 1024**3)
            with self.assertRaisesRegex(MemoryError, 'capped at 8 GiB'):
                check_allocation(9 * 1024**3, 'above processor cap', limit=8 * 1024**3)

    def test_parent_cgroup_limit_is_respected(self):
        values = {'/proc/self/cgroup': '0::/user.slice/job.scope\n',
                  '/sys/fs/cgroup/user.slice/job.scope/memory.max': 'max',
                  '/sys/fs/cgroup/user.slice/job.scope/memory.current': '100',
                  '/sys/fs/cgroup/user.slice/memory.max': '1000',
                  '/sys/fs/cgroup/user.slice/memory.current': '700'}
        def read(path, *args, **kwargs):
            if str(path) not in values:
                raise FileNotFoundError(str(path))
            return values[str(path)]
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=5000)), patch.object(Path, 'read_text', read):
            self.assertEqual(available_bytes(), 300)

    def test_chunked_processor_checks_budget_before_merge_only_when_enabled(self):
        from src.data.preprocess.core import PreProcessor
        from src.data.util import DataBlock

        def block(date):
            return DataBlock(np.ones((1, 1, 1, 1), dtype=np.float32), secid=[1], date=[date], feature=['value'])

        for cap in (1, None):
            with self.subTest(cap=cap):
                processor = create_autospec(PreProcessor, instance=True)
                processor.configure_mock(
                    key='budget_test', load_start=20201231, load_end=20210104,
                    DateChunkYears=1, CalculationWindow=1, ChunkFillNan=None,
                    MaxAllocationBytes=cap, MemTrace=False,
                    logger=Mock(timer=lambda *args, **kwargs: nullcontext()),
                    load_dump=Mock(return_value=DataBlock()),
                    pre_process=Mock(side_effect=[block(20201231), block(20210104)]),
                )
                with patch('src.data.util.memory_guard.available_bytes', return_value=1000) as available, patch.object(
                    DataBlock, 'merge', wraps=DataBlock.merge
                ) as merge:
                    if cap is not None:
                        with self.assertRaisesRegex(MemoryError, 'year merge'):
                            PreProcessor.load_with_extension(processor)
                        merge.assert_not_called()
                    else:
                        result = PreProcessor.load_with_extension(processor)
                        merge.assert_called()
                        available.assert_not_called()
                        self.assertEqual(result.date.tolist(), [20201231, 20210104])


if __name__ == '__main__':
    unittest.main()
