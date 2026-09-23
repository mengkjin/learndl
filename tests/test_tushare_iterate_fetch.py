"""Paginated Tushare fetches must resume past a page cap without skipping a page."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.data.download.tushare.basic.fetcher import IterateFetchPageLimit , TushareIterateFetcher
from src.data.download.tushare.task.t05_mutual_fund import FundPortfolioFetcher


def _pages(n_pages: int , limit: int):
    """Return a callable that yields ``n_pages`` frames, then an empty frame."""
    def api(**kwargs) -> pd.DataFrame:
        offset = int(kwargs['offset'])
        page = offset // limit
        if page >= n_pages:
            return pd.DataFrame()
        return pd.DataFrame({'page': [offset] , 'value': [page]})
    return api


class IterateFetchResumeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)

    def _fetcher(self , api , * , limit: int = 10 , max_fetch_times: int = 3 , breakpoint: bool = True) -> TushareIterateFetcher:
        fetcher = TushareIterateFetcher(
            'IterateFetchTest' , api , limit ,
            max_fetch_times = max_fetch_times , breakpoint = breakpoint ,
            vb_level = 'never' , period = '20260630')
        fetcher.cache_root = self.tmp
        fetcher.breakpoint_path = self.tmp / 'bkpt'
        return fetcher

    def test_resume_keeps_every_page(self):
        limit , n_pages , cap = 10 , 7 , 3
        api = _pages(n_pages , limit)
        first = self._fetcher(api , limit = limit , max_fetch_times = cap)
        with self.assertRaises(IterateFetchPageLimit):
            first.fetch()

        # Reproduce the old handler, which stored next_offset one page too far.
        metadata_path = first.metadata_path
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata['next_offset'] , cap * limit)
        metadata['next_offset'] = (cap + 1) * limit
        metadata_path.write_text(json.dumps(metadata))

        frames: list[pd.DataFrame] = []
        fetcher = self._fetcher(api , limit = limit , max_fetch_times = cap)
        for _ in range(5):
            try:
                frames.append(fetcher.fetch())
                break
            except IterateFetchPageLimit:
                fetcher = self._fetcher(api , limit = limit , max_fetch_times = cap)
        else:
            self.fail('paginated fetch did not finish')
        result = pd.concat(frames , ignore_index = True)
        self.assertEqual(result['page'].tolist() , [i * limit for i in range(n_pages)])
        self.assertFalse(fetcher.breakpoint_path.exists())

    def test_api_error_retries_the_failed_offset(self):
        limit = 10

        def api(**kwargs) -> pd.DataFrame:
            offset = int(kwargs['offset'])
            if offset == 20 and not api.allow_20:
                raise RuntimeError('最多访问')
            if offset >= 30:
                return pd.DataFrame()
            return pd.DataFrame({'page': [offset]})
        api.allow_20 = False

        fetcher = self._fetcher(api , limit = limit , max_fetch_times = 10)
        with self.assertRaises(RuntimeError):
            fetcher.fetch()

        api.allow_20 = True
        result = self._fetcher(api , limit = limit , max_fetch_times = 10).fetch()
        self.assertEqual(result['page'].tolist() , [0 , 10 , 20])

    def test_cap_without_breakpoint_still_raises(self):
        fetcher = self._fetcher(_pages(5 , 10) , breakpoint = False)
        with self.assertRaises(IterateFetchPageLimit):
            fetcher.fetch()
        self.assertFalse(fetcher.breakpoint_path.exists())

    def test_expired_breakpoint_and_stale_cache_are_removed(self):
        calls: list[int] = []

        def api(**kwargs) -> pd.DataFrame:
            offset = int(kwargs['offset'])
            calls.append(offset)
            if offset >= 10:
                return pd.DataFrame()
            return pd.DataFrame({'page': [offset]})

        fetcher = self._fetcher(api , limit = 10 , max_fetch_times = 1)
        with self.assertRaises(IterateFetchPageLimit):
            fetcher.fetch()
        self.assertTrue(list(fetcher.breakpoint_path.glob('bkpt.*.feather')))

        metadata = json.loads(fetcher.metadata_path.read_text())
        metadata['expiration_date'] = 0
        fetcher.metadata_path.write_text(json.dumps(metadata))

        stale = self.tmp / 'OldFetcher' / 'old_api' / 'period=20200101'
        stale.mkdir(parents = True)
        (stale / 'metadata.json').write_text(json.dumps({
            'expiration_date': 0 , 'next_offset': 10 , 'breakpoints': [0] ,
        }))
        (stale / 'bkpt.0.feather').write_bytes(b'x')

        orphan = self.tmp / 'Orphan' / 'api' / 'pages'
        orphan.mkdir(parents = True)
        orphan_file = orphan / 'bkpt.0.feather'
        orphan_file.write_bytes(b'x')
        old = os.path.getmtime(orphan_file) - 5 * 3600
        os.utime(orphan_file , (old , old))

        calls.clear()
        again = self._fetcher(api , limit = 10 , max_fetch_times = 1)
        with self.assertRaises(IterateFetchPageLimit):
            again.fetch()
        self.assertEqual(calls[0] , 0)
        self.assertFalse(stale.exists())
        self.assertFalse((self.tmp / 'OldFetcher').exists())
        self.assertFalse((self.tmp / 'Orphan').exists())
        self.assertTrue(again.breakpoint_path.joinpath('bkpt.0.feather').exists())


class FundPortfolioPageCapTest(unittest.TestCase):
    def test_get_data_continues_after_page_cap(self):
        inst = FundPortfolioFetcher(vb_level = 'never')
        done = pd.DataFrame({
            'symbol': ['000001.SZ'] ,
            'ann_date': [20260630] ,
            'end_date': [20260630] ,
        })
        inst.iterate_fetch = Mock(side_effect = [IterateFetchPageLimit('got more than 500 dfs') , done])
        out = inst.get_data(20260630)
        self.assertEqual(inst.iterate_fetch.call_count , 2)
        self.assertIn('secid' , out.columns)
        self.assertFalse(out.empty)


if __name__ == '__main__':
    unittest.main()
