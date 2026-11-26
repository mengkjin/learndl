from contextlib import closing
from datetime import datetime
from pathlib import Path
import signal
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import torch

from src.automation import nightly_maintenance
from src.automation.scheduling.nightly_window import latest_nightly_slot, load_window
from src.automation.scheduling import runtime
from src.automation.scheduling.calendar_lanes import eligibility
from src.data.download.other_source.rcquant import sec_native_revision as native
from src.data.download.other_source.rcquant import sec_none_migration as migration
from src.data.preprocess.nightly_rebuild import rebuild_block
from src.data.preprocess.processors.trade import (
    PrePro_15m, PrePro_15mt, PrePro_30m, PrePro_30mt, standardize_num_trades,
)
from src.data.util import DataBlock
from src.proj.db.io.mmap import ArrayMemoryMap
from tests.test_rcquant_sec_none import minute_frame


class NightlyTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        db = self.enterContext(closing(sqlite3.connect(':memory:')))
        self.job = migration.Migration(self.root, self.root / 'exfactor', db, None)

    def test_window_uses_scheduled_date_even_after_midnight(self):
        zone = ZoneInfo('Asia/Shanghai')
        slot = datetime(2026, 11, 26, 21, tzinfo=zone).timestamp()
        window = load_window(Path(__file__).resolve().parents[1], slot)
        self.assertEqual(latest_nightly_slot(slot + 4 * 3600), slot)
        self.assertEqual(datetime.fromtimestamp(window.cutoff, zone), datetime(2026, 11, 27, tzinfo=zone))
        self.assertEqual(window.hard_deadline - window.cutoff, 600)
        run = {'started_at': slot, 'timeout_seconds': 3*3600, 'hard_deadline': window.hard_deadline}
        self.assertFalse(runtime.timeout_due(run, window.cutoff))
        self.assertTrue(runtime.timeout_due(run, window.hard_deadline))

    def test_delayed_calendar_request_expires_at_configured_cutoff(self):
        scheduled = datetime(2026, 11, 26, 21, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        directory = self.root / 'configs/automation'
        directory.mkdir(parents=True)
        (directory / 'nightly.yaml').write_text(
            "timezone: Asia/Shanghai\ncutoff: '01:30'\ngrace_minutes: 10\ntasks: [rcquant_minutes]\n"
        )
        cutoff = scheduled + 4.5 * 3600
        with patch('src.automation.scheduling.calendar_lanes.ROOT', self.root):
            for name in ('nightly_maintenance', 'rcquant_sec_none'):
                config = {'timezone': 'Asia/Shanghai', 'tasks': {name: {'enabled': True, 'entrypoint': name}}}
                key = f'{name}:{int(scheduled // 60)}'
                self.assertEqual(eligibility(name, key, config, cutoff - 1)[0], 'ready')
                self.assertEqual(eligibility(name, key, config, cutoff)[0], 'expired')

    def test_supervisor_force_stops_only_the_expired_nightly_session(self):
        store = Mock()
        run = {'id': 'nightly', 'phase': 'running', 'hard_deadline': 1000}
        store.active.return_value = [run, {'id': 'unrelated'}]
        worker = object()
        with patch.object(runtime, 'members', return_value=[worker]), \
                patch.object(runtime, 'signal_members') as terminate:
            runtime.inspect_timeouts(store, now=1000, run_id='nightly')
        self.assertEqual(run['term_at'], 1000)
        self.assertEqual(run['kill_at'], 1000)
        terminate.assert_called_once_with(run, [worker], signal.SIGKILL)

    def test_trade_count_transform_has_no_lookahead_and_preserves_missing(self):
        counts = torch.arange(1., 46.).reshape(1, -1, 1)
        counts[:, 6] = 0
        counts[:, 14] = torch.nan
        whole = standardize_num_trades(counts)
        self.assertTrue(whole[:, :10].isnan().all())
        self.assertTrue(whole[:, 14].isnan().all())
        torch.testing.assert_close(whole[:, 30:35], standardize_num_trades(counts[:, 10:35])[:, -5:])
        changed = counts.clone()
        changed[:, 35:] = 5000
        torch.testing.assert_close(whole[:, :35], standardize_num_trades(changed)[:, :35], equal_nan=True)
        self.assertEqual(standardize_num_trades(torch.zeros_like(counts))[0, 20, 0].item(), 0.)

    def test_trade_count_blocks_preserve_the_six_existing_features(self):
        dates = np.arange(25) + 20240101
        ids = np.array([1])
        for key, period, old, new in (
            ('15m', 16, PrePro_15m, PrePro_15mt), ('30m', 8, PrePro_30m, PrePro_30mt),
        ):
            with self.subTest(key=key):
                prices = torch.full((1, 25, period, 7), 10.)
                prices[..., 6] = torch.arange(25).reshape(1, -1, 1)
                bars = DataBlock(prices, ids, dates, np.array([*migration.PRICE, 'volume', 'vwap', 'num_trades']))
                daily = DataBlock(torch.ones(1, 25, 1, 2), ids, dates, np.array(['turn_fl', 'preclose']))
                baseline = old('predict').process_blocks({key: bars.copy(), 'day': daily.copy()})
                processor = new('predict')
                actual = processor.process_blocks({key: bars.copy(), 'day': daily.copy()})
                self.assertEqual(actual.feature.tolist(), [*baseline.feature.tolist(), 'num_trades'])
                torch.testing.assert_close(actual.values[..., :6], baseline.values)
                self.assertTrue(actual.values[:, -1, :, -1].isfinite().all())
                missing = processor.process_blocks({key: baseline.align_feature(bars.feature[:-1]),
                                                    'day': daily.copy()})
                self.assertTrue(missing.values[..., -1].isnan().all())

    def test_old_completed_day_still_downloads_once_and_resumes_partial_publish(self):
        date = native.DATE
        path = self.job.minute_path(date)
        path.parent.mkdir()
        minute_frame(date, price=8.).to_feather(path)
        self.job.mark(self.job.repair_stage('done'), date)
        response = minute_frame(date).assign(order_book_id='000001.XSHE')
        fetch = Mock(return_value=response)
        instruments = pd.DataFrame({'code': ['000001.XSHE'], 'is_active': [True]})
        def descriptor(_, key):
            return SimpleNamespace(
                path_exact=lambda day: self.root / key / f'{day}.feather', clear_date_cache=lambda: None,
            )
        original_publish = self.job.publish_files

        def interrupt(day, prepared, **kwargs):
            original_publish(day, {'outputs': prepared['outputs'][:2]}, **kwargs)
            raise migration.EveningEnded

        with patch.object(self.job, 'direct_none', fetch), \
                patch.object(migration.bar_min, 'rcquant_instrument_list', return_value=instruments), \
                patch.object(migration, 'DBPath', side_effect=descriptor), \
                patch.object(native.DB, 'load_pl', return_value=pl.DataFrame()):
            with patch.object(self.job, 'publish_files', side_effect=interrupt), self.assertRaises(migration.EveningEnded):
                native.repair_native_day(self.job)
            self.assertIsNone(self.job.read(f'{native.REVISION}_done', date))
            self.assertTrue(native.repair_native_day(self.job))
            self.assertTrue(native.repair_native_day(self.job))
            fetch.assert_called_once_with(date, ['000001.XSHE'])
            done = self.job.require_record(f'{native.REVISION}_done', date)
            self.assertEqual(len(done['outputs']), 7)
            for output in done['outputs']:
                self.assertTrue(migration.file_matches(Path(output['target']), output['digest']))
                self.assertEqual(pl.read_ipc(output['target'], memory_map=False)['num_trades'].sum(),
                                 response['num_trades'].sum())

    def test_receipt_missing_after_batch_write_reuses_download(self):
        response = pl.from_pandas(minute_frame(native.DATE).assign(order_book_id='000001.XSHE'))
        response = response.select(native.ID, 'datetime', *native.QUOTE)
        path = self.root / 'download' / '0000.feather'
        migration.write_frame(response, path)
        self.job.mark(f'{native.REVISION}_batch', 0, {'digest': None})
        self.job.direct_none = Mock(side_effect=AssertionError('already downloaded'))
        expected = response.select(native.ID, 'datetime')
        result = native._download(self.job, self.root, {'ids': ['000001.XSHE'], 'batches': [['000001.XSHE']]},
                                  expected)
        self.assertEqual(result.height, 240)

    def test_partial_native_response_only_retries_missing_stock(self):
        ids = ['000001.XSHE', '000002.XSHE']
        responses = [minute_frame(native.DATE).assign(order_book_id=stock) for stock in ids]
        expected = pl.from_pandas(pd.concat(responses)).select(native.ID, 'datetime')
        self.job.direct_none = Mock(side_effect=responses)
        manifest = {'ids': ids, 'batches': [ids]}
        self.assertIsNone(native._download(self.job, self.root, manifest, expected))
        result = native._download(self.job, self.root, manifest, expected)
        self.assertEqual(result.height, 480)
        self.assertEqual([call.args for call in self.job.direct_none.call_args_list],
                         [(native.DATE, ids), (native.DATE, ids[1:])])

    def test_missing_daily_volume_is_excluded_from_mismatch_counts(self):
        old = pl.from_pandas(minute_frame(native.DATE))
        stock = '000001.XSHE'
        bars = old.with_columns(pl.lit(stock).alias(native.ID))
        mapping = {old['code'][0]: stock}
        stocks = self.job.stock_mapping(mapping)
        day = stocks.select('secid').with_columns(
            *[pl.lit(10.).alias(name) for name in migration.PRICE],
            pl.lit(float('nan')).alias('volume'),
        )
        with patch.object(native.DB, 'load_pl', return_value=day):
            counts = native._audit(self.root, old, bars, stocks, mapping)
        self.assertEqual(counts['native_volume_mismatches'], 0)
        result = pl.read_ipc(self.root / 'daily_comparison.feather')
        self.assertEqual(result['volume_mismatch_transition'].item(), 'no_reference')

    def test_dispatch_holds_downstream_until_native_and_history_publish(self):
        handler = nightly_maintenance.rcquant_minutes
        with patch.object(migration.bar_min, 'RQ_PATH', self.root), \
                patch.object(native, 'repair_native_day', return_value=False), \
                patch.object(migration.Migration, 'run') as history, \
                patch('src.data.download.other_source.rcquant.sec_minute_downstream.rebuild_minute_consumers') as downstream:
            self.assertEqual(handler(lambda: None, 0)[0], 'waiting on data')
            history.assert_not_called()
            downstream.assert_not_called()
        with patch.object(migration.bar_min, 'RQ_PATH', self.root), \
                patch.object(native, 'repair_native_day', return_value=True), \
                patch.object(migration.Migration, 'run', return_value=None) as history, \
                patch('src.data.download.other_source.rcquant.sec_minute_downstream.rebuild_minute_consumers') as downstream:
            self.assertEqual(handler(lambda: None, 0)[0], 'waiting on data')
            history.assert_called_once_with(revised_dates=(native.DATE,), rebuild=False)
            downstream.assert_not_called()

    def test_production_exemption_requires_active_daily_only_inputs(self):
        from src.data.download.other_source.rcquant.sec_minute_downstream import _review_production
        strategy = self.root / 'configs/strategy'
        strategy.mkdir(parents=True)
        (strategy / 'model.yaml').write_text('prediction: {daily: {name: gru@gru_day}}')
        (strategy / 'trading_port.yaml').write_text('trading_ports: {live: {alpha: daily}}')
        path = self.root / 'models/nn/gru@gru_day/configs/production.yaml'
        path.parent.mkdir(parents=True)
        with patch('src.proj.PATH', SimpleNamespace(main=self.root)):
            with self.assertRaisesRegex(RuntimeError, 'gru@gru_day'):
                _review_production([])
            path.write_text('model: {input.type: combo, input.data.types: day}')
            with self.assertRaisesRegex(RuntimeError, 'gru@gru_day'):
                _review_production([])
            path.write_text('model: {input.type: data, input.data.types: day}')
            self.assertEqual(_review_production([])['models'], ['gru@gru_day'])

    def test_datablock_reuses_shards_after_deadline(self):
        from src.data.preprocess import nightly_rebuild
        target = self.root / 'X_mock.mmap'
        dates = [20250131, 20250203]
        def preprocess(start, end):
            ids = np.array([1, 2] if start == dates[0] else [2, 3])
            return DataBlock(torch.ones(2, 1, 1, 1), ids, np.array([start]), np.array(['close']))
        processor = SimpleNamespace(load_start=dates[0], load_end=dates[-1], pre_process=Mock(side_effect=preprocess))
        with patch.object(nightly_rebuild.PrePros, 'get_processor', return_value=processor), \
                patch.object(nightly_rebuild.DataBlock, 'path_preprocess', return_value=target), \
                patch.object(nightly_rebuild, 'Dates', return_value=dates):
            self.job.guard = Mock(side_effect=[None, migration.EveningEnded])
            with self.assertRaises(migration.EveningEnded):
                rebuild_block(self.job, 'test', 'predict', 'mock')
            self.job.guard = Mock()
            rebuild_block(self.job, 'test', 'predict', 'mock')
        self.assertEqual(processor.pre_process.call_count, 2)
        result = ArrayMemoryMap.load(target, index=True)
        self.assertEqual(result['index']['secid'].tolist(), [1, 2, 3])
        expected = torch.tensor([[1., torch.nan], [1., 1.], [torch.nan, 1.]])
        torch.testing.assert_close(result['values'][:, :, 0, 0], expected, equal_nan=True)

    def test_resume_copy_after_new_date_and_feature(self):
        from src.data.preprocess import nightly_rebuild
        target = self.root / 'X_factor.mmap'
        dates = [20250131, 20250228, 20250303]
        def preprocess(start, end):
            features = ['first'] if start == dates[0] else ['new', 'first']
            row = [1.] if start == dates[0] else [2., 3.]
            return DataBlock(torch.tensor(row).reshape(1, 1, 1, -1), np.array([1]),
                             np.array([start]), np.array(features))
        processor = SimpleNamespace(load_start=dates[0], load_end=dates[1], pre_process=Mock(side_effect=preprocess))
        def until(start, end):
            return [d for d in dates if start <= d <= end]
        with patch.object(nightly_rebuild.PrePros, 'get_processor', return_value=processor), \
                patch.object(DataBlock, 'path_preprocess', return_value=target), \
                patch.object(nightly_rebuild, 'Dates', side_effect=until):
            self.job.guard = Mock(side_effect=[None, None, None, migration.EveningEnded])
            with self.assertRaises(migration.EveningEnded):
                rebuild_block(self.job, 'test', 'fit', 'mock')
            self.assertTrue(self.job.read('test_block', 'fit/mock/copy/202501-20250131'))
            self.job.guard = Mock()
            processor.load_end = dates[-1]
            rebuild_block(self.job, 'test', 'fit', 'mock')
        self.assertEqual(processor.pre_process.call_count, 3)
        result = ArrayMemoryMap.load(target, index=True)
        self.assertEqual(result['index']['date'].tolist(), dates)
        self.assertEqual(result['index']['feature'].tolist(), ['first', 'new'])
        expected = torch.tensor([[1., torch.nan], [3., 2.], [3., 2.]])
        torch.testing.assert_close(result['values'][0, :, 0, :], expected, equal_nan=True)

    def test_minchars_fill_and_listing_mask_match_full_merge(self):
        from src.data.preprocess import nightly_rebuild
        from src.data.util.stock_info import INFO
        dates = [20250131, 20250228]
        def preprocess(start, end):
            ids = np.array([1, 2] if start == dates[0] else [1, 3])
            return DataBlock(torch.tensor([4., torch.nan]).reshape(2, 1, 1, 1), ids,
                             np.array([start]), np.array(['feature']))
        processor = SimpleNamespace(load_start=dates[0], load_end=dates[-1], pre_process=preprocess,
                                    ChunkFillNan=0., mask={'list_dt': 1})
        info = pd.DataFrame({'secid': [1, 2, 3], 'list_dt': [20240102, 20240102, 20250210],
                             'delist_dt': [-1, -1, -1]})
        target = self.root / 'X_minc.mmap'
        with patch.object(nightly_rebuild.PrePros, 'get_processor', return_value=processor), \
                patch.object(DataBlock, 'path_preprocess', return_value=target), \
                patch.object(nightly_rebuild, 'Dates', return_value=dates), \
                patch.object(INFO, 'get_desc', return_value=info):
            expected = DataBlock.merge([preprocess(d, d) for d in dates]).fillna(0.).mask_values(processor.mask)
            rebuild_block(self.job, 'test', 'predict', 'minc')
        result = ArrayMemoryMap.load(target)['values']
        torch.testing.assert_close(result, expected.values, equal_nan=True)
