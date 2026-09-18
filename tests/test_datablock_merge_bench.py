"""Benchmark ``DataBlock.merge`` mesh vs broadcast on minc-shaped panels.

Shapes follow production ``PrePro_minc``:

- Selected columns from ``selected_by_db_key()`` (daily + roll + tag).
- ``DateChunkYears=1`` year concat: disjoint dates, growing secid universe.
- Inner year job: three tables stacked on feature (same secid/date).
- Incremental overlay: ``ExtentionOverlay=10`` overlapping dates.

Unittest (tiny, CI-safe)::

    python -m unittest discover -s tests -p test_datablock_merge_bench.py

Server bench::

    uv run python tests/test_datablock_merge_bench.py
    uv run python tests/test_datablock_merge_bench.py --profile minc --repeats 3

``--max-mesh-gb`` skips the mesh path when ``intersect_mesh_bytes`` would exceed
that many GiB (default 8).  Raise it on a large RAM box if you want mesh at
2021-scale too.
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.data.preprocess.mem_trace import fmt_bytes , rss_bytes
from src.data.util import DataBlock
from src.func.basic import INTERSECT_MESH_MAX_BYTES , IntersectCopyMethod , index_merge , intersect_mesh_bytes

# Production merge kwargs from ``PreProcessor.load_with_extension`` year chunks.
_YEAR_MERGE : dict[str , Any] = {
    'inplace' : False ,
    'secid_method' : 'union' ,
    'date_method' : 'union' ,
    'inday_method' : 'check' ,
    'feature_method' : 'stack' ,
}

# A-share-like listing growth used when not overridden by --n.
_LISTED_2010 = 1800
_LISTED_STEP = 250
_LISTED_CAP = 5500
_TD_PER_YEAR = 242
_OVERLAY_BARS = 10


@dataclass(frozen = True)
class Case:
    """One production-shaped merge to time."""

    name : str
    blocks : tuple[DataBlock , ...]
    note : str


def listed_count(year : int) -> int:
    """Approximate listed-A count at ``year`` (2010≈1800, +250/yr, cap 5500)."""
    return min(_LISTED_CAP , _LISTED_2010 + (year - 2010) * _LISTED_STEP)


def feature_counts() -> dict[str , int]:
    """Selected min_chars column counts by db_key."""
    from src.data.update.custom.min_chars._catalog import selected_by_db_key
    grouped = selected_by_db_key()
    return {key : len(cols) for key , cols in grouped.items()}


def _block(n : int , dates : np.ndarray , names : list[str] , secid0 : int = 1) -> DataBlock:
    """Dense float32 panel ``(N, T, 1, F)`` with integer secids and unique feature names."""
    secid = np.arange(secid0 , secid0 + n , dtype = np.int64)
    feature = np.array(names)
    values = torch.randn(n , len(dates) , 1 , len(names) , dtype = torch.float32)
    return DataBlock(values , secid = secid , date = dates , feature = feature)


def _year_dates(year : int , n_td : int) -> np.ndarray:
    """Synthetic yyyymmdd-like dates unique per calendar year (not a real calendar)."""
    return year * 10000 + 101 + np.arange(n_td , dtype = np.int64)


def _feat_names(prefix : str , n : int) -> list[str]:
    return [f'{prefix}{i:04d}' for i in range(n)]


def _all_feature_names(counts : dict[str , int]) -> list[str]:
    names : list[str] = []
    for prefix , n in counts.items():
        names.extend(_feat_names(prefix , n))
    return names


def case_year_concat(
    acc_years : int ,
    * ,
    start_year : int = 2010 ,
    n_td : int = _TD_PER_YEAR ,
    n_scale : float = 1.0 ,
    counts : dict[str , int] | None = None ,
) -> Case:
    """
    Merge ``acc_years`` of history with the next calendar year.

    Mirrors ``load_with_extension`` after ``DateChunkYears=1``: disjoint dates,
    stacked identical features, union secid (new listings).
    """
    counts = counts or feature_counts()
    names = _all_feature_names(counts)
    y0 = start_year
    y1 = start_year + acc_years - 1
    y_new = y1 + 1
    n_acc = max(1 , int(listed_count(y1) * n_scale))
    n_new = max(1 , int(listed_count(y_new) * n_scale))
    acc_dates = np.concatenate([_year_dates(y , n_td) for y in range(y0 , y1 + 1)])
    prev = _block(n_acc , acc_dates , names)
    year = _block(n_new , _year_dates(y_new , n_td) , names)
    return Case(
        name = f'year-concat {y0}-{y1}+{y_new}' ,
        blocks = (prev , year) ,
        note = f'N={n_acc}+{n_new} T={len(acc_dates)}+{n_td} F={len(names)}' ,
    )


def case_table_stack(
    year : int = 2021 ,
    * ,
    n_td : int = _TD_PER_YEAR ,
    n_scale : float = 1.0 ,
    counts : dict[str , int] | None = None ,
) -> Case:
    """Merge the three min_chars tables for one year (feature stack, same secid/date)."""
    counts = counts or feature_counts()
    n = max(1 , int(listed_count(year) * n_scale))
    dates = _year_dates(year , n_td)
    blocks = tuple(
        _block(n , dates , _feat_names(key , n_feat))
        for key , n_feat in counts.items()
    )
    f_str = '+'.join(str(v) for v in counts.values())
    return Case(
        name = f'table-stack {year}' ,
        blocks = blocks ,
        note = f'N={n} T={n_td} F={f_str}' ,
    )


def case_date_overlay(
    year : int = 2021 ,
    * ,
    n_td : int = _TD_PER_YEAR ,
    overlay : int = _OVERLAY_BARS ,
    n_scale : float = 1.0 ,
    counts : dict[str , int] | None = None ,
) -> Case:
    """Incremental extend: last ``overlay`` dates overlap, then new trading days."""
    counts = counts or feature_counts()
    names = _all_feature_names(counts)
    n = max(1 , int(listed_count(year) * n_scale))
    dumped = _year_dates(year , n_td)
    extra = year * 10000 + 800 + np.arange(n_td , dtype = np.int64)
    new_dates = np.concatenate([dumped[-overlay:] , extra])
    return Case(
        name = f'date-overlay {year} o={overlay}' ,
        blocks = (_block(n , dumped , names) , _block(n , new_dates , names)) ,
        note = f'N={n} T={n_td}+{overlay + n_td} F={len(names)}' ,
    )


def _say(msg : str) -> None:
    print(msg , flush = True)


def _mesh_peak(blocks : tuple[DataBlock , ...]) -> int:
    """Max ``intersect_mesh_bytes`` over sources vs the union axes (same as merge)."""
    secid = index_merge([b.secid for b in blocks] , method = 'union')
    date = index_merge([b.date for b in blocks] , method = 'union')
    inday = index_merge([b.inday for b in blocks] , method = 'check')
    feature = index_merge([b.feature for b in blocks] , method = 'stack')
    dst = [secid , date , inday , feature]
    return max(
        intersect_mesh_bytes(dst , [b.secid , b.date , b.inday , b.feature])
        for b in blocks
    )


def _time_merge(
    blocks : tuple[DataBlock , ...] ,
    copy_method : IntersectCopyMethod ,
    repeats : int ,
    warmup : int ,
) -> float:
    """Median wall time in ms of ``DataBlock.merge``. Inputs are not mutated (inplace=False)."""

    def once() -> None:
        DataBlock.merge(list(blocks) , copy_method = copy_method , **_YEAR_MERGE)

    for _ in range(warmup):
        once()
        gc.collect()
    samples : list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        once()
        samples.append((time.perf_counter() - t0) * 1000.0)
        gc.collect()
    return statistics.median(samples)


def _run_case(
    case : Case ,
    * ,
    repeats : int ,
    warmup : int ,
    max_mesh_bytes : int ,
) -> None:
    mesh_b = _mesh_peak(case.blocks)
    union_n = len(np.unique(np.concatenate([b.secid for b in case.blocks])))
    union_t = len(np.unique(np.concatenate([b.date for b in case.blocks])))
    union_f = len(list(dict.fromkeys(nm for b in case.blocks for nm in b.feature.tolist())))
    union_b = union_n * union_t * union_f * 4
    auto = 'mesh' if mesh_b <= INTERSECT_MESH_MAX_BYTES else 'broadcast'
    rss0 = rss_bytes()
    _say(
        f'{case.name}  {case.note}  union={fmt_bytes(union_b)}{union_n, union_t, 1, union_f}  '
        f'mesh4d={fmt_bytes(mesh_b)}  auto={auto}  rss={fmt_bytes(rss0)}' ,
    )
    methods : tuple[IntersectCopyMethod , ...] = ('mesh' , 'broadcast')
    for method in methods:
        if method == 'mesh' and mesh_b > max_mesh_bytes:
            _say(
                f'  {method:<10} skipped (mesh4d {fmt_bytes(mesh_b)} > --max-mesh-gb {fmt_bytes(max_mesh_bytes)})' ,
            )
            continue
        try:
            ms = _time_merge(case.blocks , method , repeats = repeats , warmup = warmup)
        except (MemoryError , RuntimeError) as exc:
            _say(f'  {method:<10} FAIL {type(exc).__name__}: {exc}')
            gc.collect()
            continue
        _say(
            f'  {method:<10} {ms:8.1f} ms   rss={fmt_bytes(rss_bytes())}' ,
        )


def _cases_for_profile(profile : str , n_scale : float) -> list[Case]:
    counts = feature_counts()
    n_feat = sum(counts.values())
    _say(
        f'minc selected F={n_feat} by table {counts}  profile={profile}  n_scale={n_scale}' ,
    )
    if profile == 'smoke':
        n_scale = min(n_scale , 0.05)
        return [
            case_table_stack(2012 , n_td = 40 , n_scale = n_scale , counts = counts) ,
            case_year_concat(1 , n_td = 40 , n_scale = n_scale , counts = counts) ,
            case_date_overlay(2012 , n_td = 40 , overlay = 5 , n_scale = n_scale , counts = counts) ,
        ]
    if profile == 'server':
        return [
            case_table_stack(2021 , n_scale = n_scale , counts = counts) ,
            case_year_concat(1 , n_scale = n_scale , counts = counts) ,
            case_year_concat(5 , n_scale = n_scale , counts = counts) ,
            case_year_concat(11 , n_scale = n_scale , counts = counts) ,
            case_date_overlay(2021 , n_scale = n_scale , counts = counts) ,
        ]
    if profile == 'minc':
        return [
            case_table_stack(2010 , n_scale = n_scale , counts = counts) ,
            case_table_stack(2021 , n_scale = n_scale , counts = counts) ,
            case_year_concat(1 , n_scale = n_scale , counts = counts) ,
            case_year_concat(5 , n_scale = n_scale , counts = counts) ,
            case_year_concat(11 , n_scale = n_scale , counts = counts) ,
            case_year_concat(15 , n_scale = n_scale , counts = counts) ,
            case_date_overlay(2021 , n_scale = n_scale , counts = counts) ,
        ]
    raise ValueError(f'unknown profile {profile!r}')


def parse_args(argv : list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description = 'DataBlock.merge mesh vs broadcast bench (minc-shaped)')
    p.add_argument(
        '--profile' , choices = ('smoke' , 'server' , 'minc') , default = 'server' ,
        help = 'smoke=tiny; server=2021 tables + concat 1/5/11y; minc=full 2010-2025 concat steps' ,
    )
    p.add_argument('--repeats' , type = int , default = 3)
    p.add_argument('--warmup' , type = int , default = 1)
    p.add_argument(
        '--max-mesh-gb' , type = float , default = 8.0 ,
        help = 'skip mesh when index grids exceed this many GiB' ,
    )
    p.add_argument(
        '--n-scale' , type = float , default = 1.0 ,
        help = 'scale listed-count (1.0 ≈ production N; 0.2 for a RAM-limited box)' ,
    )
    return p.parse_args(argv)


def main(argv : list[str] | None = None) -> None:
    args = parse_args(argv)
    max_mesh_bytes = int(args.max_mesh_gb * 1024 ** 3)
    _say(
        f'DataBlock.merge bench  cutoff(auto)={fmt_bytes(INTERSECT_MESH_MAX_BYTES)}  '
        f'max_mesh={fmt_bytes(max_mesh_bytes)}  repeats={args.repeats} warmup={args.warmup}  '
        f'rss={fmt_bytes(rss_bytes())}' ,
    )
    for case in _cases_for_profile(args.profile , args.n_scale):
        _run_case(
            case , repeats = args.repeats , warmup = args.warmup , max_mesh_bytes = max_mesh_bytes ,
        )
        for blk in case.blocks:
            blk.uninitiate()
        gc.collect()
    _say(f'done rss={fmt_bytes(rss_bytes())}')


class TestMergeCopyAgreement(unittest.TestCase):
    """Tiny correctness check so unittest discovery stays cheap."""

    def test_mesh_and_broadcast_match_on_year_concat(self) -> None:
        case = case_year_concat(1 , n_td = 20 , n_scale = 0.02)
        mesh = DataBlock.merge(list(case.blocks) , copy_method = 'mesh' , **_YEAR_MERGE)
        bcast = DataBlock.merge(list(case.blocks) , copy_method = 'broadcast' , **_YEAR_MERGE)
        self.assertEqual(tuple(mesh.shape) , tuple(bcast.shape))
        self.assertTrue(torch.equal(mesh.values.isnan() , bcast.values.isnan()))
        self.assertTrue(torch.allclose(mesh.values.nan_to_num() , bcast.values.nan_to_num()))

    def test_mesh_and_broadcast_match_on_table_stack(self) -> None:
        case = case_table_stack(2011 , n_td = 16 , n_scale = 0.02)
        m_mesh = DataBlock.merge(list(case.blocks) , copy_method = 'mesh' , **_YEAR_MERGE)
        m_bc = DataBlock.merge(list(case.blocks) , copy_method = 'broadcast' , **_YEAR_MERGE)
        self.assertEqual(tuple(m_mesh.shape) , tuple(m_bc.shape))
        self.assertTrue(torch.allclose(m_mesh.values.nan_to_num() , m_bc.values.nan_to_num()))


if __name__ == '__main__':
    main()
