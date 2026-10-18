"""Raw volume survives preprocessing; empty datasets fail before any training."""
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.data import DataBlock
from src.data.preprocess.processors.trade import PrePro_15mraw
from src.res.model.util.data.batch_input_loader import DataloaderParam
from src.res.model.util.data.data_module import DataModule
from src.res.model.util.data.operations import DataOperator
from src.res.model.util.data.prenorm import SpecialDataPrenorm
from src.res.model.util.metric.stage_metrics import EpochMetrics


class Raw15mTest(TestCase):
    def test_volume_survives_and_complete_windows_remain_eligible(self):
        features = ['close', 'high', 'low', 'open', 'volume', 'vwap']
        values = torch.arange(1, 2 * 30 * 16 * 6 + 1, dtype=torch.float32).reshape(2, 30, 16, 6)
        raw = DataBlock(values, np.array([1, 2]), np.arange(30), np.array(features))
        processor = object.__new__(PrePro_15mraw)
        block = processor.process_blocks({'15m': raw})
        self.assertEqual(list(block.feature), ['open', 'close', 'high', 'low', 'vwap', 'volume'])
        torch.testing.assert_close(block.values[..., -1], values[..., features.index('volume')])
        torch.testing.assert_close(block.values[..., 0], values[..., features.index('open')])

        config = SimpleNamespace(module_type='nn', seq_steps={'15mraw': 1}, input_verify_x_max_nan_ratio=.1)
        operator = DataOperator(config, DataloaderParam(seqlens={'15mraw': 30}))
        indices = torch.tensor([29])
        self.assertTrue(operator.finite_position('15mraw', block.values, indices).all())
        legacy = block.align_feature(['open', 'close', 'high', 'low', 'vwap', 'turn_fl'])
        self.assertFalse(operator.finite_position('15mraw', legacy.values, indices).any())
        normalized = SpecialDataPrenorm.sp_15mraw(block.values.clone(), features=block.feature)
        self.assertTrue(torch.isfinite(normalized).all())
        torch.testing.assert_close(normalized[..., -1].sum((1, 2)), torch.ones(2))


class EmptyFitTest(TestCase):
    def data(self, stage, counts):
        data = DataModule(config=SimpleNamespace(input_keys_all={'data': ['15mraw']}, module_type='nn'))
        data.loader_param = DataloaderParam(stage=stage, model_date=20170103, seqlens={'15mraw': 30})
        data.loader_dict.update({key: [object()] * count for key, count in counts.items()})
        return data

    def test_empty_train_or_validation_fails_during_setup_including_cached_loaders(self):
        for counts in ({'train': 0, 'valid': 0}, {'train': 1, 'valid': 0}, {'train': 0, 'valid': 1}):
            for rebuild in (True, False):
                with self.subTest(counts=counts, rebuild=rebuild):
                    data = self.data('fit', counts)
                    with patch.object(data, 'setup_new_param', return_value=rebuild), \
                         patch.object(data, 'setup_loader_kwargs'), \
                         patch.object(data, 'setup_loader_inputs'), \
                         patch.object(data, 'setup_loader_static'), \
                         self.assertRaisesRegex(ValueError, 'No usable fit batches.*20170103.*15mraw'):
                        data.setup('fit')

    def test_nonempty_fit_and_empty_prediction_are_allowed(self):
        for stage, counts in [('fit', {'train': 1, 'valid': 1}), ('predict', {'predict': 0})]:
            data = self.data(stage, counts)
            with patch.object(data, 'setup_new_param', return_value=False):
                data.setup(stage)

    def test_empty_metrics_raise_explanatory_error_and_normal_metrics_are_unchanged(self):
        metrics = EpochMetrics(Mock())
        metrics.new(dataset='train')
        metrics.tables['accuracies'] = []
        with self.assertRaisesRegex(ValueError, 'No batch metrics.*train'):
            metrics.accuracies_dict
        metrics.tables['accuracies'] = [pd.DataFrame({'score': [.2, .4]},
            index=pd.MultiIndex.from_tuples([(0,), (1,)], names=['batch']))]
        self.assertAlmostEqual(metrics.accuracies_dict['score'], .3)


if __name__ == '__main__':
    main()
