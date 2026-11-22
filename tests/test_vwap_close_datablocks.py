"""VWAP repair and rolling-week alignment in the shipping processors."""
from unittest import TestCase, main

import numpy as np
import torch

from src.data import DataBlock
from src.data.preprocess.core import TradePreProcessor
from src.data.preprocess.processors.trade import (
    PrePro_15m, PrePro_30m, PrePro_60m, PrePro_30mcont, PrePro_week,
)

FEATURES = ['open', 'close', 'high', 'low', 'vwap', 'turn_fl']
BARS = {'15m': 16, '30m': 8, '60m': 4, '30mcont': 8, 'week': 5}
PROCESSORS = {'15m': PrePro_15m, '30m': PrePro_30m, '60m': PrePro_60m,
              '30mcont': PrePro_30mcont, 'week': PrePro_week}


def source_blocks(key):
    dates = np.arange(20240101, 20240109)
    if key == 'week':
        features = ['adjfactor', 'preclose', *FEATURES]
        values = torch.ones(2, 8, 1, 8, dtype=torch.float64)
        values[..., 0] = 2.
        values[..., 1] = 10.
        values[..., 2:7] = 11.
        values[:, :, 0, 3] = torch.arange(12., 20.)  # each day's close differs
        return {'day': DataBlock(values, [1, 2], dates, features)}
    features = ['close', 'high', 'low', 'open', 'volume', 'vwap']
    values = torch.full((2, 8, BARS[key], 6), 11., dtype=torch.float64)
    values[..., 0] = 12.
    values[..., 4] = 100.
    day = torch.tensor([10., 1.], dtype=torch.float64).reshape(1, 1, 1, 2).repeat(2, 8, 1, 1)
    return {key if key != '30mcont' else '30m': DataBlock(values, [1, 2], dates, features),
            'day': DataBlock(day, [1, 2], dates, ['preclose', 'turn_fl'])}


def clone_blocks(blocks):
    return {key: DataBlock(block.values.clone(), block.secid.copy(), block.date.copy(), block.feature.copy())
            for key, block in blocks.items()}


class VwapCloseTest(TestCase):
    def test_all_five_processors_match_same_source_bar_close_replacement(self):
        for key, cls in PROCESSORS.items():
            with self.subTest(key=key):
                blocks = source_blocks(key)
                raw = next(iter(blocks.values()))
                v, c = [list(raw.feature).index(f) for f in ('vwap', 'close')]
                raw.values[0, 3, 0, v] = torch.inf
                raw.values[1, 4, 0, v] = -torch.inf
                expected_inputs = clone_blocks(blocks)
                expected_raw = next(iter(expected_inputs.values()))
                mask = torch.isinf(expected_raw.values[..., v])
                expected_raw.values[..., v][mask] = expected_raw.values[..., c][mask]
                proc = object.__new__(cls)
                expected = TradePreProcessor.process_blocks(proc, expected_inputs)
                records = []
                proc.vwap_inf_repair_observer = records.append
                actual = proc.process_blocks(blocks)
                torch.testing.assert_close(actual.values, expected.values, equal_nan=True)
                self.assertFalse(torch.isinf(actual.values[..., 4]).any())
                self.assertEqual(sum(r['inf_count'] for r in records), 2)
                if key == 'week':
                    # Day 4 appears at different inday positions in later windows.
                    for end_day in range(4, 8):
                        bar = 4 - (end_day - 4)
                        self.assertEqual(actual.values[1, end_day, bar, 4], actual.values[1, end_day, bar, 1])

    def test_only_infinite_vwap_changes_including_invalid_close(self):
        values = torch.ones(1, 1, 6, 6)
        values[0, 0, :, 4] = torch.tensor([torch.inf, -torch.inf, torch.nan, 7., torch.inf, -torch.inf])
        values[0, 0, :, 1] = torch.tensor([2., 3., 4., 5., torch.nan, torch.inf])
        values[..., 2] = torch.inf  # Other fields must not be sanitized.
        block = DataBlock(values.clone(), [1], [20240102], FEATURES)
        object.__new__(PrePro_15m)._repair_vwap_inf(block, 'test')
        expected = values.clone()
        expected[0, 0, :, 4] = torch.tensor([2., 3., torch.nan, 7., torch.nan, torch.nan])
        torch.testing.assert_close(block.values, expected, equal_nan=True)

    def test_float32_overflow_is_repaired_after_normalisation(self):
        for key, cls in PROCESSORS.items():
            with self.subTest(key=key):
                blocks = source_blocks(key)
                raw = next(iter(blocks.values()))
                raw.values[0, 4, 0, list(raw.feature).index('vwap')] = 1e40
                result = object.__new__(cls).process_blocks(blocks)
                bar = 4 if key == 'week' else 0
                self.assertEqual(result.values[0, 4, bar, 4], result.values[0, 4, bar, 1])
                self.assertFalse(torch.isinf(result.values[..., 4]).any())


if __name__ == '__main__':
    main()
