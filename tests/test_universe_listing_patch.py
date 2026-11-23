"""Regression coverage for historical universe membership and its migration."""
from __future__ import annotations

from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.res.factor.util.classes import universe as univ

__all__ = []


class UniverseListingTest(unittest.TestCase):
    def test_filter_before_top_cutoff(self):
        date = 20200601
        desc = pd.DataFrame(index=[2, 3])
        values = pd.DataFrame({'circ_mv': [100, 50, 10], 'secid': [920344, 2, 3]})
        with (patch.object(univ.DATAVENDOR.INFO, 'get_desc', return_value=desc) as get_desc,
              patch.object(univ.DATAVENDOR.TRADE, 'closest_date', return_value=date),
              patch.object(univ.DATAVENDOR.TRADE, 'get_val', return_value=values),
              patch.object(univ.UniverseExclusions, 'append_exclusions', side_effect=lambda df, d: df),
              patch.object(univ.Save, 'df'), patch.object(univ.DB, 'path')):
            for name, expected in [('all', [2, 3]), ('top1', [2])]:
                actual = univ.Universe(name).get_universe_df(date, rebuild=True)
                self.assertEqual(actual.secid.tolist(), expected)
                get_desc.assert_called_with(date)

    def test_patch_preserves_surviving_rows_and_clean_files(self):
        migration = runpy.run_path(str(Path(__file__).resolve().parents[1] /
            'scripts/.version_patch/0.7.9/001_universe_listing_dates.py'))
        desc = pd.DataFrame({'secid': [920344, 920344, 920809, 2],
                             'list_dt': [20180101, 20200602, 20200602, 20200601],
                             'delist_dt': [20200601, 99991231, 99991231, 99991231]})
        with tempfile.TemporaryDirectory() as tmp, patch.object(univ.DB, 'load', return_value=desc):
            ctx = SimpleNamespace(root=Path(tmp), backup_dir=Path(tmp)/'backup')
            archive = ctx.root/'data/Export/universe'
            folder = archive/'top-1000/2020'
            folder.mkdir(parents=True)
            first = folder/'top-1000.20200601.feather'
            second = folder/'top-1000.20200602.feather'
            frame = pd.DataFrame({'secid': [837344, 2, 920809, 999], 'st': [False, True, True, False]})
            frame.to_feather(first)
            frame.to_feather(second)
            original = first.read_bytes()
            stamp = second.stat().st_mtime_ns
            with self.assertRaisesRegex(ValueError, 'Inactive securities'):
                migration['verify'](ctx)
            migration['apply'](ctx)
            self.assertTrue(migration['verify'](ctx))
            pd.testing.assert_frame_equal(pd.read_feather(first), frame.iloc[[1, 3]].reset_index(drop=True))
            self.assertEqual((ctx.backup_dir/first.relative_to(archive)).read_bytes(), original)
            self.assertEqual(second.stat().st_mtime_ns, stamp)
            stamp = first.stat().st_mtime_ns
            migration['apply'](ctx)
            self.assertEqual(first.stat().st_mtime_ns, stamp)
