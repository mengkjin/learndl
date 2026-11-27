from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from src.data.download.other_source.rcquant import bar_min
from src.data.loader.min_kline import MKLINE
from src.data.util.transform import trade_min_reform, trade_min_reform_pl
from src.proj import DB
from src.proj.db.io.snapshot import ReadSnapshot


def minute_frame(last=239, *, counts=True):
    minute = np.arange(last - 239, last + 1)
    df = pd.DataFrame(dict(secid=1, minute=minute, open=10., high=12., low=8.,
                           close=10., amount=100., volume=10., vwap=10.))
    df.loc[:14, ['amount', 'volume']] = 0
    df.loc[7, ['open', 'high', 'low', 'vwap']] = np.nan
    if counts:
        df['num_trades'] = 10
    return df


class StockMinutePolarsTest(TestCase):
    def test_compatibility_readers_and_alternative_source_for_all_periods(self):
        date = 20250422
        with TemporaryDirectory() as work:
            root = Path(work)
            with patch('src.proj.db.interface.db_path.PATH', SimpleNamespace(db=root / 'db')):
                source = minute_frame()
                for key in ('min', '5min', '10min', '15min', '30min', '60min'):
                    with self.subTest(key=key):
                        path = DB.path('trade_js', key, date)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        source.to_feather(path, compression='lz4')
                        with patch.object(pd, 'read_feather', side_effect=AssertionError('pandas read')), \
                             patch.object(pl, 'read_ipc', wraps=pl.read_ipc) as read:
                            pd.testing.assert_frame_equal(DB.load('trade_ts', key, date, use_alt=True), source)
                            pd.testing.assert_frame_equal(DB.load('trade_js', key, date), source)
                            pd.testing.assert_frame_equal(DB.loads('trade_ts', key, dates=[date], use_alt=True),
                                                          source.assign(date=date)[['date', *source.columns]])
                            pd.testing.assert_frame_equal(DB.load_pl('trade_ts', key, date, use_alt=True).to_pandas(),
                                                          source)
                            self.assertEqual(read.call_count, 4)
                            self.assertTrue(DB.load('trade_ts', key, date).empty)
                            with self.assertRaises(FileNotFoundError):
                                DB.load('trade_ts', key, date, missing_ok=False)
                with closing(ReadSnapshot(root / 'snapshot', root / 'db')) as snapshot, snapshot.activate():
                    pd.testing.assert_frame_equal(DB.load('trade_js', 'min', date), source)
                    path = DB.path('trade_js', 'min', date)
                    source.assign(close=123.).to_feather(path)
                    pd.testing.assert_frame_equal(DB.load('trade_js', 'min', date), source)
                    self.assertEqual(DB.loads('trade_js', 'min', dates=[date])['close'].iloc[0], 10.)

    def test_batch_reads_mixed_codecs_and_maps_security_ids(self):
        with TemporaryDirectory() as work, \
             patch('src.proj.db.interface.db_path.PATH', SimpleNamespace(db=Path(work))), \
             patch('src.proj.db.basic.code_mapper.MACHINE.dfdata.get', return_value={'839729': 920729}):
            dates = [20250418, 20250421, 20250422]
            for date, codec in zip(dates, ('uncompressed', 'lz4', 'zstd')):
                path = DB.path('trade_ts', 'min', date)
                path.parent.mkdir(parents=True, exist_ok=True)
                pl.from_pandas(minute_frame().assign(secid=839729)).write_ipc(path, compression=codec)
            with patch.object(pd, 'read_feather', side_effect=AssertionError('pandas read')), \
                 patch.object(pl, 'read_ipc', wraps=pl.read_ipc) as read:
                pandas = DB.loads('trade_ts', 'min', dates=dates)
                polars = DB.loads_pl('trade_ts', 'min', dates=dates).to_pandas()
                self.assertEqual(read.call_count, 6)
            self.assertEqual(len(pandas), 720)
            self.assertEqual(set(pandas['secid']), {920729})
            self.assertEqual(set(pandas['date']), set(dates))
            cols = ['date', 'secid', 'minute']
            pd.testing.assert_frame_equal(pandas.sort_values(cols).reset_index(drop=True),
                                          polars[pandas.columns].sort_values(cols).reset_index(drop=True))

    def test_mkline_reads_once_and_uses_5min_on_empty_1min(self):
        MKLINE.clear_all()
        self.addCleanup(MKLINE.clear_all)
        with patch.object(DB, 'load_pl', side_effect=[pl.DataFrame(), pl.from_pandas(minute_frame())]) as read, \
             patch.object(MKLINE, 'get_secid', return_value=np.array([1])):
            for _ in range(2):
                self.assertEqual(MKLINE.get_kline(20250422).height, 240)
        self.assertEqual([call.args[1] for call in read.call_args_list], ['min', '5min'])

    def test_polars_reform_matches_pandas_for_numbering_order_and_missing_values(self):
        for last in (239, 240):
            for counts in (True, False):
                source = minute_frame(last, counts=counts).sample(frac=1, random_state=2)
                for period in (5, 10, 15, 30, 60):
                    with self.subTest(last=last, counts=counts, period=period):
                        old = trade_min_reform(source.copy(), period)
                        new = trade_min_reform_pl(pl.from_pandas(source), period).to_pandas()
                        pd.testing.assert_frame_equal(new, old, check_exact=True)

    def test_rcquant_multiple_periods_read_source_once(self):
        downloader = bar_min.RcquantMinBarDownloader()
        with patch.object(bar_min, 'x_mins_to_update', return_value=[5, 30, 60]), \
             patch.object(DB, 'load_pl', return_value=pl.from_pandas(minute_frame())) as read, \
             patch.object(bar_min, '_save_bar') as save:
            downloader._publish_missing_periods(20250422, bar_min.MinDataType.SEC)
        read.assert_called_once()
        self.assertEqual([call.args[2] for call in save.call_args_list], ['5min', '30min', '60min'])
