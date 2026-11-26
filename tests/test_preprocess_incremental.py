"""Weekly fit updates extend existing dumps and revisit maturing forward returns."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from src.data.preprocess.processors.labels import PrePro_y
from src.data.preprocess.task import PreProcessorTask, PrePros
from src.data.util import DataBlock
from src.proj import CALENDAR


class IncrementalFitTest(TestCase):
    def test_weekly_update_appends_tail_and_refreshes_pending_labels(self):
        dates = CALENDAR.range(20240102, 20240208)
        source = torch.arange(2 * len(dates), dtype=torch.float32).reshape(1, -1, 1, 2)
        old = source[:, :-3].clone()
        old[:, -18:, :, 1] = torch.nan
        old[:, 0] = -7
        dump = DataBlock(old, [1], dates[:-3], ['short', 'long'])
        processor = PrePro_y('fit')
        processor.load_start, processor.load_end = int(dates[0]), int(dates[-1])

        def fetch(start, end, **kwargs):
            mask = (dates >= start) & (dates <= end)
            return DataBlock(source[:, mask].clone(), [1], dates[mask], ['short', 'long'])

        with patch.object(PrePros, 'get_processor', return_value=processor), \
             patch.object(processor, 'should_be_skipped', return_value=False), \
             patch.object(processor, 'load_dump', return_value=dump), \
             patch.object(processor, 'pre_process', side_effect=fetch) as queried, \
             patch.object(processor, 'save_dump') as saved, patch.object(processor, 'save_norm'):
            PreProcessorTask.proceed_update(frame='fit', data_types=['y'], confirm=False)
            result = saved.call_args.args[0]
            self.assertEqual(queried.call_count, 1)
            self.assertGreater(queried.call_args.args[0], dates[0])
            np.testing.assert_array_equal(result.date, dates)
            torch.testing.assert_close(result.values[:, :1], old[:, :1])
            torch.testing.assert_close(result.values[:, 1:], source[:, 1:])
            with patch.object(processor, 'load_dump', return_value=result):
                PreProcessorTask.proceed_update(frame='fit', data_types=['y'], confirm=False)
            self.assertEqual(queried.call_count, 1)


    def test_incremental_build_keeps_dates_added_since_processor_creation(self):
        dates = np.array([20261124, 20261125, 20261126])
        processor = PrePro_y('predict')
        processor.load_start, processor.load_end = dates[0], dates[1]
        dump = DataBlock(torch.ones(1, 3, 1, 1), [1], dates, ['feature'])
        with patch.object(processor, 'load_dump', return_value=dump):
            result = processor.load_with_extension()
        np.testing.assert_array_equal(result.date, dates)

    def test_both_extends_stale_fit_dump_without_waiting_for_weekly(self):
        from src.res.model.util.data.module_data import ModuleData

        self.enterContext(patch.object(CALENDAR, 'update_to', return_value=20261124))
        dates = CALENDAR.range(20261026, 20261124)
        old_dates = dates[dates <= 20261109]
        field = ['std_lag1_10']
        dump = DataBlock(torch.ones(1, len(old_dates), 1, 1), [1], old_dates, field)
        processor = PrePro_y('fit')
        def extension(start, end, **kwargs):
            selected = dates[(dates >= start) & (dates <= end)]
            return DataBlock(torch.ones(1, len(selected), 1, 1), [1], selected, field)

        with (
            patch.object(CALENDAR, 'updated', return_value=20261124),
            patch.object(DataBlock, 'max_data_date', return_value=20261109),
            patch.object(PrePros, 'get_processor', return_value=processor),
            patch.object(processor, 'load_dump', return_value=dump),
            patch.object(processor, 'pre_process', side_effect=extension) as compute,
        ):
            fit = ModuleData.initialize(['day'], field, 'fit')
            both = ModuleData.initialize(['day'], field, 'both')
            self.assertEqual(fit.target_start_end()[1], 20261109)
            self.assertEqual(both.target_start_end()[1], 20261124)
            block = both.load_one('y', dates=dates)
            self.assertEqual(compute.call_args.args[1], 20261124)
            np.testing.assert_array_equal(block.date, dates)
            self.assertFalse(processor.enable_saving)
