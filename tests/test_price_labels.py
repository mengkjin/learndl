"""Open/VWAP returns always start on the trading day after the label date."""
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.proj import CALENDAR, Dates
from src.data import DataBlock
from src.data.update.custom.labels import ClassicLabelsUpdater, calc_price_labels
from src.data.preprocess.processors import PrePro_y, PrePro_y_lag0, PrePro_y_open, PrePro_y_vwap
from src.data.preprocess.task import DATASET_FIT, DATASET_PREDICT
from src.res.model.util.label_spec import LabelSpec
from src.res.model.util.config import ModelConfig
from src.res.model.util.data.operations import DataOperator


class PriceLabelsTest(TestCase):
    def test_fixed_next_trading_day_and_holding_period(self):
        # Friday -> Monday: calendar-day arithmetic would incorrectly use Saturday.
        for price in ('open', 'vwap'):
            for days in (5, 10, 20):
                with self.subTest(price=price, days=days):
                    expected_start = CALENDAR.td(20240105, 1).as_int()
                    expected_end = CALENDAR.td(expected_start, days).as_int()
                    raw = pd.DataFrame({'ret': [0.1, np.nan]}, index=pd.Index([1, 2], name='secid'))
                    with patch('src.data.update.custom.labels.get_period_ret', return_value=raw) as read:
                        result = calc_price_labels(20240105, days, price)
                    read.assert_called_once_with(expected_start, expected_end, price)
                    self.assertEqual(expected_start, 20240108)
                    self.assertEqual(list(result.columns), ['secid', f'rtn_{days}_{price}'])
                    self.assertAlmostEqual(result.iloc[0, 1], 0.1)
                    self.assertTrue(pd.isna(result.iloc[1, 1]))

    def test_empty_and_invalid_requests(self):
        for raw in (None, pd.DataFrame()):
            with patch('src.data.update.custom.labels.get_period_ret', return_value=raw):
                self.assertIsNone(calc_price_labels(20240102, 10, 'open'))
        for days, price in [(0, 'open'), (-1, 'vwap'), (10, 'close')]:
            with self.assertRaises(ValueError):
                calc_price_labels(20240102, days, price)
        with self.assertRaisesRegex(ValueError, 'next trading day'):
            ClassicLabelsUpdater.update_one(20240102, 10, False, 'ret10_open', price_type='open')

    def test_incremental_jobs_skip_existing_and_overwrite(self):
        for existing, overwrite, expected in [(False, False, 13), (True, False, 0), (True, True, 13)]:
            with patch.object(ClassicLabelsUpdater, 'START_DATE', 20240102), patch('src.data.update.custom.labels.CALENDAR.updated', return_value=20240229), patch('src.data.update.custom.labels.DB.dates', return_value=Dates([20240102]) if existing else Dates()), patch.object(ClassicLabelsUpdater, 'update_one') as update:
                ClassicLabelsUpdater.proceed_update(start=20240102, end=20240102, overwrite=overwrite)
                self.assertEqual(update.call_count, expected)
                if expected:
                    jobs = {call.args[3]: (call.args[2], call.kwargs['price_type']) for call in update.call_args_list}
                    for price in ('open', 'vwap'):
                        for days in (5, 10, 20):
                            self.assertEqual(jobs[f'ret{days}_{price}'], (True, price))
                    self.assertEqual(jobs['ret10_lag'], (True, 'close'))
                    self.assertEqual(jobs['ret10'], (False, 'close'))
        with patch.object(ClassicLabelsUpdater, 'START_DATE', 20240102), patch('src.data.update.custom.labels.CALENDAR.updated', return_value=20240103), patch('src.data.update.custom.labels.DB.dates', return_value=Dates()), patch.object(ClassicLabelsUpdater, 'update_one') as update:
            ClassicLabelsUpdater.proceed_update(start=20240102, end=20240102)
            update.assert_not_called()

    def test_update_writes_new_key_without_close_residuals(self):
        for price in ('open', 'vwap'):
            raw = pd.DataFrame({'secid': [1], f'rtn_10_{price}': [0.1]})
            with patch('src.data.update.custom.labels.calc_price_labels', return_value=raw) as calc, patch('src.data.update.custom.labels.DB.save') as save, patch('src.data.update.custom.labels.calc_classic_labels', side_effect=AssertionError('No close residuals')):
                ClassicLabelsUpdater.update_one(20240102, 10, True, f'ret10_{price}', price_type=price)
                calc.assert_called_once_with(20240102, 10, price)
                self.assertEqual(save.call_args.args[1:], ('labels_ts', f'ret10_{price}', 20240102))

    def test_preprocessor_transforms_match_existing_labels(self):
        torch.manual_seed(7)
        dates, stocks = np.array([20240102, 20240103]), np.arange(50)
        values = torch.randn(50, 2, 1, 1)
        risk = DataBlock(torch.randn(50, 2, 1, 3), stocks, dates, ['ind1', 'ind2', 'size'])
        expected = PrePro_y_lag0('fit').process({'y': DataBlock(values.clone(), stocks, dates, ['rtn_lag0_10']), 'risk': risk})
        for price, cls in [('open', PrePro_y_open), ('vwap', PrePro_y_vwap)]:
            result = cls('fit').process({'y': DataBlock(values.clone(), stocks, dates, [f'rtn_10_{price}']), 'risk': risk})
            self.assertEqual(list(result.feature), [f'rtn_10_{price}', f'std_10_{price}'])
            torch.testing.assert_close(result.values, expected.values)
            self.assertIn(f'y_{price}', DATASET_FIT)
            self.assertIn(f'y_{price}', DATASET_PREDICT)
            with patch('src.data.preprocess.processors.labels.BlockLoader') as loader:
                cls('fit').block_loaders()
                self.assertEqual(loader.call_args_list[0].args, ('labels_ts', [f'ret10_{price}', f'ret20_{price}', f'ret5_{price}']))
        # Legacy short-term fields in default y retain their original behavior.
        legacy = PrePro_y('fit').process({'y': DataBlock(values.clone(), stocks, dates, ['rtn_lag1_3_open']), 'risk': risk})
        self.assertEqual(list(legacy.feature), ['rtn_lag1_3_open'])

    def test_horizon_used_by_astgnn_and_zero_fill(self):
        for price in ('open', 'vwap'):
            name = f'y_{price}@std_10_{price}'
            self.assertEqual(LabelSpec.parse(name).horizon, (1, 10))
            self.assertEqual(DataOperator.parse_label_horizon(name), (1, 10))
            config = SimpleNamespace(sample_method='sequential', labels=[name, 'std_lag1_5'], skip_horizon=12)
            ModelConfig.check_astgnn_config(config)
            config.skip_horizon = 11
            with self.assertRaises(ValueError):
                ModelConfig.check_astgnn_config(config)
            op = DataOperator(SimpleNamespace())
            with patch('src.data.loader.RISK.get_res', return_value=pd.DataFrame({'secid': [1]})) as residual:
                result = op.fill_y_zero_no_trade(torch.full((1, 1, 1), torch.nan), np.array([20240105]), np.array([1]), [name])
            residual.assert_called_once_with(CALENDAR.td(20240105, 11).as_int())
            self.assertEqual(result.item(), 0.)


if __name__ == '__main__':
    main()
