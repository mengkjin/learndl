"""Trace real numeric failure modes without reading or writing server data."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import DataBlock
from src.data.util.transform import trade_min_reform

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/.autofix/repair_15mraw_inf.py'
spec = importlib.util.spec_from_file_location('repair_15mraw_inf', SCRIPT)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def source_row(**kwargs):
    return dict(secid=1, minute=0, open=2., close=2., high=2., low=2.,
                vwap=np.inf, amount=2., volume=0.) | kwargs


class InfAuditTest(TestCase):
    def audit(self, rows, values=None, error=None):
        values = values if values is not None else torch.ones(1, 1, 16, 6)
        if values.isfinite().all():
            values[0, 0, 0, 4] = torch.inf
        block = DataBlock(values, [1], [20240102], task.repair.EXPECTED_FEATURES)
        before = block.values.clone()
        item = {}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(task, 'load_source_day', return_value=(pd.DataFrame(rows), '/raw/15min/20240102.feather')), \
             patch.object(task.Logger, 'note'):
            if error:
                with self.assertRaisesRegex(ValueError, error):
                    task.audit_and_sanitize(block, SimpleNamespace(frame='fit'), item, Path(directory))
            else:
                task.audit_and_sanitize(block, SimpleNamespace(frame='fit'), item, Path(directory))
                torch.testing.assert_close(block.values[~torch.isinf(before)], before[~torch.isinf(before)], equal_nan=True)
                self.assertFalse(torch.isinf(block.values).any())
            records = [json.loads(line) for line in (Path(directory) / 'fit_inf_evidence.jsonl').read_text().splitlines()]
        return item['inf_audit'], records, block

    def test_zero_volume_stored_inf_is_explained_and_converted(self):
        stats, records, block = self.audit([source_row()])
        self.assertEqual(stats['status'], 'verified')
        self.assertEqual(stats['converted_to_nan'], 1)
        self.assertEqual(records[0]['origin'], 'stored_raw_inf')
        self.assertEqual(records[0]['mechanism'], 'nonzero_amount_div_zero_volume_reproduces_inf')
        self.assertEqual(records[0]['raw_context']['volume'], 0.)
        self.assertTrue(torch.isnan(block.values[0, 0, 0, 4]))

    def test_signed_inf_and_existing_nan_are_preserved_as_missing_only(self):
        values = torch.ones(1, 1, 16, 6)
        values[0, 0, 0, 4] = -torch.inf
        values[0, 0, 2, 3] = torch.nan
        _, records, _ = self.audit([source_row(vwap=-np.inf, amount=-2.)], values)
        self.assertEqual(records[0]['source_value'], '-inf')

    def test_finite_raw_float32_overflow_is_identified(self):
        stats, records, _ = self.audit([source_row(vwap=1e40, amount=1e40, volume=1.)])
        self.assertEqual(stats['by_origin'], {'float32_cast_overflow': 1})
        self.assertEqual(records[0]['source_value'], 1e40)

    def test_stored_vwap_overflow_and_nonfinite_operand_evidence(self):
        cases = [(source_row(amount=1e38, volume=1e-10), 'float32_division_or_cast_overflow_reproduces_inf'),
                 (source_row(amount=np.inf, volume=1.), 'nonfinite_operand_reproduces_inf')]
        for row, mechanism in cases:
            with self.subTest(mechanism=mechanism):
                _, records, _ = self.audit([row])
                self.assertEqual(records[0]['mechanism'], mechanism)

    def test_unknown_upstream_origin_is_explicit_not_falsely_divide_by_zero(self):
        _, records, _ = self.audit([source_row(amount=2., volume=1.)])
        self.assertEqual(records[0]['origin'], 'stored_raw_inf')
        self.assertEqual(records[0]['mechanism'], 'upstream_origin_unresolved')

    def test_unexplained_finite_or_nan_to_inf_stops(self):
        for raw in (2., np.nan, -np.inf):
            with self.subTest(raw=raw):
                stats, records, _ = self.audit([source_row(vwap=raw)], error='Unexplained')
                self.assertEqual(stats['status'], 'failed')
                self.assertEqual(records[0]['origin'], 'unexplained_preprocessing_inf')

    def test_missing_conflicting_and_noncanonical_sources_stop(self):
        for rows, error in [([], 'missing source'),
                            ([source_row(minute=1)], 'missing 1/20240102/0/vwap'),
                            ([source_row(), source_row(amount=3.)], 'conflicting'),
                            ([source_row(minute=16)], 'noncanonical')]:
            with self.subTest(error=error):
                stats, _, _ = self.audit(rows, error=error)
                self.assertEqual(stats['status'], 'failed')

    def test_every_inf_has_evidence_not_only_sampled_cells(self):
        values = torch.ones(1, 1, 16, 6)
        values[0, 0, :, 4] = torch.inf
        rows = [source_row(minute=i) for i in range(16)]
        stats, records, _ = self.audit(rows, values)
        self.assertEqual(stats['inf_count'], 16)
        self.assertEqual(len(records), 16)
        self.assertEqual(len(stats['examples']), 1)
        self.assertEqual({r['bar'] for r in records}, set(range(16)))

    def test_real_preprocessor_reproduces_source_inf_and_cast_overflow(self):
        import polars as pl
        from src.data.preprocess.processors.trade import PrePro_15mraw
        rows = [source_row(minute=i, vwap=2., volume=1., date=20240102) for i in range(16)]
        rows[0].update(vwap=np.inf, volume=0.)
        rows[1]['high'] = 1e40
        rows[2]['low'] = np.nan
        raw = DataBlock.from_polars(pl.DataFrame(rows))
        proc = object.__new__(PrePro_15mraw)
        block = proc.process_blocks({'15m': raw})
        self.assertEqual(block.values.dtype, torch.float32)
        stats, records, _ = self.audit(rows, block.values)
        self.assertEqual(stats['by_origin'], {'stored_raw_inf': 1, 'float32_cast_overflow': 1})
        self.assertEqual({r['feature'] for r in records}, {'vwap', 'high'})

    def test_source_loader_matches_production_fallback_selection(self):
        import polars as pl
        with patch.object(task.DB, 'DBPath') as path, patch.object(task.DB, 'loads_pl', return_value=pl.DataFrame([source_row()])) as load:
            path.return_value.path.return_value = Path('/alternative/15min/20240102.feather')
            _, actual = task.load_source_day(20240102)
        self.assertEqual(actual, '/alternative/15min/20240102.feather')
        path.return_value.path.assert_called_once_with(20240102, use_alt=True)
        load.assert_called_once_with('trade_ts', '15min', dates=[20240102], use_alt=True, vb_level='never')

    def test_request_only_after_success_with_new_id_and_no_dry_run_writes(self):
        with patch.object(task.repair, 'repair_datablocks') as repair, patch.object(task.repair, 'request_training') as train:
            task.run_task(dry_run=True)
            train.assert_not_called()
            task.run_task()
            train.assert_called_once_with(task.REQUEST_ID)
            repair.side_effect = ValueError('audit failed')
            with self.assertRaisesRegex(ValueError, 'audit failed'):
                task.run_task()
            train.assert_called_once()


class MinuteAggregationTest(TestCase):
    def bars(self, amount=4., volume=2.):
        return pd.DataFrame(dict(secid=1, minute=np.arange(240), open=2., close=2.,
                                 high=2., low=2., vwap=2., amount=amount, volume=volume))

    def test_correct_formula_for_normal_positive_volume(self):
        result = trade_min_reform(self.bars(), 15)
        np.testing.assert_allclose(result['vwap'], 2.)
        np.testing.assert_allclose(result['amount'], 60.)
        np.testing.assert_allclose(result['volume'], 30.)

    def test_zero_negative_or_nonfinite_operands_never_generate_inf_vwap(self):
        for amount, volume in [(4., 0.), (0., 0.), (4., -1.), (np.inf, 2.), (4., np.inf), (1e308, 1e-308)]:
            with self.subTest(amount=amount, volume=volume):
                with np.errstate(over='ignore', invalid='ignore'):
                    result = trade_min_reform(self.bars(amount, volume), 15)
                self.assertTrue(result['vwap'].isna().all())

    def test_float32_group_sum_uses_float64_before_overflow(self):
        bars = self.bars(1e38, 1e38).astype({'amount': 'float32', 'volume': 'float32'})
        result = trade_min_reform(bars, 15)
        self.assertTrue(np.isfinite(result[['amount', 'volume', 'vwap']]).all().all())
        np.testing.assert_allclose(result['vwap'], 1.)


if __name__ == '__main__':
    main()
