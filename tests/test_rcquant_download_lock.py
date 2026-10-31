"""Concurrent provider calls, partial caches and atomic publication failure."""
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

import pandas as pd

from src.data.download.other_source.rcquant import bar_min
from src.data.download.other_source.rcquant.storage import atomic_frame, sample_size


def download(updater, event, first_n=-1):
    event.wait(5)
    assert bar_min.RcquantMinBarDownloader.rcquant_bar_min(updater, 20261029, 'sec', first_n)


class DownloadLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.enterContext(patch.object(bar_min, 'RQ_PATH', self.root))
        self.enterContext(patch.object(bar_min.RQInitializer, 'init', return_value=True))
        self.enterContext(patch.object(bar_min, 'rcquant_instrument_list', return_value=pd.DataFrame({'code':['000001.XSHE'], 'is_active':[True]})))
        self.enterContext(patch.object(bar_min, '_save_bar', return_value=True))
        self.updater = SimpleNamespace(indent=0, vb_level=0, rcquant_min_to_normal_min=lambda df, kind: df)
        self.calls = self.root/'calls'
        def provider(*args, **kwargs):
            with self.calls.open('a') as stream:
                stream.write('request\n')
            return pd.DataFrame({'order_book_id':['000001.XSHE'], 'datetime':[pd.Timestamp('2026-10-29 09:31')], 'total_turnover':[1.]})
        self.enterContext(patch.object(bar_min, 'rq_get_price', side_effect=provider))

    def test_two_processes_download_once_and_reuse_after_conversion_failure(self):
        context = multiprocessing.get_context('fork')
        event = context.Event()
        jobs = [context.Process(target=download, args=(self.updater,event)) for _ in range(2)]
        for job in jobs:
            job.start()
        event.set()
        for job in jobs:
            job.join(15)
            if job.is_alive():
                job.kill()
                job.join()
            self.assertEqual(job.exitcode, 0)
        self.assertEqual(self.calls.read_text().splitlines(), ['request'])
        self.assertTrue(bar_min.RcquantMinBarDownloader.rcquant_bar_min(self.updater, 20261029, 'sec'))
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)

    def test_sample_cache_does_not_satisfy_full_download(self):
        method = bar_min.RcquantMinBarDownloader.rcquant_bar_min
        self.assertTrue(method(self.updater, 20261029, 'sec', 1))
        self.assertFalse((self.root/'secmin/20261029.feather').exists())
        self.assertTrue((self.root/'.samples/1/secmin/20261029.feather').exists())
        self.assertTrue(method(self.updater, 20261029, 'sec'))
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)
        self.assertEqual(sample_size.get(), -1)

    def test_failed_provider_can_retry(self):
        with patch.object(bar_min, 'rq_get_price', side_effect=RuntimeError('network')):
            with self.assertRaisesRegex(RuntimeError, 'network'):
                bar_min.RcquantMinBarDownloader.rcquant_bar_min(self.updater, 20261029, 'sec')
        self.assertTrue(bar_min.RcquantMinBarDownloader.rcquant_bar_min(self.updater, 20261029, 'sec'))

    def test_conversion_failure_reuses_raw_cache(self):
        method = bar_min.RcquantMinBarDownloader.rcquant_bar_min
        self.updater.rcquant_min_to_normal_min = Mock(side_effect=ValueError('conversion'))
        with self.assertRaisesRegex(ValueError, 'conversion'):
            method(self.updater, 20261029, 'sec')
        self.updater.rcquant_min_to_normal_min = lambda df, kind: df
        self.assertTrue(method(self.updater, 20261029, 'sec'))
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)

    def test_atomic_failure_preserves_previous_file(self):
        path = self.root/'data.feather'
        path.write_bytes(b'previous')
        def fail(frame, temp, **kwargs):
            temp.write_bytes(b'incomplete')
            raise OSError('disk full')
        with patch('src.proj.Save.df', side_effect=fail), self.assertRaises(OSError):
            atomic_frame(pd.DataFrame({'a':[1]}), path)
        self.assertEqual(path.read_bytes(), b'previous')
        self.assertEqual(list(self.root.iterdir()), [path])
