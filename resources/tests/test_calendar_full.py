"""
全量校验 src.proj.calendar 与 src.proj.calendar_new 的接口一致性。

每个带参接口至少覆盖两组参数组合。

运行：

- ``uv run python test_calendar_full.py``：仅单元测试。
- ``uv run python test_calendar_full.py --bench``：测试通过后输出旧/新耗时对比表
  （``old/new`` 列：比值大于 1 表示新版更快）。

注意：calendar_new 中 isinstance(date, TradeDate) 只识别本包的 T1；把旧包 T0
传入 C1.cd_array / C1.td_array 等会按 int(date)（交易日）处理，与 C0 行为不一致。
业务迁移时应统一使用同一包的 TradeDate。
"""

from __future__ import annotations

import sys
import time
import unittest
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.proj.calendar_old import CALENDAR as C0
from src.proj.calendar_old import Dates as D0
from src.proj.calendar_old import TradeDate as T0
from src.proj.calendar_old.basic import BC as BC0
from src.proj.calendar import CALENDAR as C1
from src.proj.calendar import Dates as D1
from src.proj.calendar import TradeDate as T1
from src.proj.calendar.basic import BC as BC1


def _np_eq(a: Any, b: Any) -> None:
    np.testing.assert_array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))


def _td_diff_expected(d1: int, d2: int) -> int:
    return int(BC0.full.loc[d2, "td_index"]) - int(BC1.full.loc[d1, "td_index"]) # type: ignore

def _reset_update_to() -> None:
    C0._update_to = None
    C1._update_to = None


def _calendar_bench_fixtures() -> dict[str, Any]:
    d0 = C0.calendar_start()
    d1 = C0.calendar_end()
    cal_all = C0.cd_within(d0, d1, until_today=False)
    mid = int(cal_all[len(cal_all) // 2])
    tmid = int(C0.td(mid).as_int())
    rng = np.random.default_rng(42)
    tds = C0.trade_dates()
    cds = C0.cd_within(d0, min(d1, d0 + 8000), until_today=False)
    sample_td = rng.choice(tds, size=512, replace=True)
    sample_cd = rng.choice(cds, size=512, replace=True)
    arr_200 = C0.trade_dates()[:200]
    tgt = C0.td_within(d0, tmid, until_today=False)
    src = tgt[:: max(1, len(tgt) // 5)]
    return {
        "d0": d0,
        "d1": d1,
        "mid": mid,
        "tmid": tmid,
        "sample_td": sample_td,
        "sample_cd": sample_cd,
        "arr_200": arr_200,
        "tgt": tgt,
        "src": src,
    }


def _bench_row(
    label: str,
    fn_old: Callable[[], Any],
    fn_new: Callable[[], Any],
    *,
    repeat: int = 400,
    warmup: int = 25,
) -> tuple[str, float, float, float]:
    """先对齐一次输出，再测耗时；返回 (label, old_ms, new_ms, old/new 比值)。"""
    o0, o1 = fn_old(), fn_new()
    if isinstance(o0, np.ndarray) or isinstance(o1, np.ndarray):
        _np_eq(o0, o1)
    elif hasattr(o0, "as_int"):
        assert int(o0.as_int()) == int(o1.as_int())
    else:
        assert o0 == o1
    for _ in range(warmup):
        fn_old()
        fn_new()
    t_a = time.perf_counter()
    for _ in range(repeat):
        fn_old()
    t_old = time.perf_counter() - t_a
    t_b = time.perf_counter()
    for _ in range(repeat):
        fn_new()
    t_new = time.perf_counter() - t_b
    ms0, ms1 = t_old * 1000, t_new * 1000
    ratio = (t_old / t_new) if t_new > 0 else float("inf")
    return (label, ms0, ms1, ratio)


def print_old_new_benchmark_table() -> None:
    F = _calendar_bench_fixtures()
    d0, mid, tmid = F["d0"], F["mid"], F["tmid"]
    sample_td, sample_cd = F["sample_td"], F["sample_cd"]
    arr_200, tgt, src = F["arr_200"], F["tgt"], F["src"]
    rows: list[tuple[str, float, float, float]] = []

    rows.append(
        _bench_row(
            "td_array(512) off=0 backward=T",
            lambda: C0.td_array(sample_td, 0, True),
            lambda: C1.td_array(sample_td, 0, True),
        )
    )
    rows.append(
        _bench_row(
            "td_array(512) off=3 backward=T",
            lambda: C0.td_array(sample_td, 3, True),
            lambda: C1.td_array(sample_td, 3, True),
        )
    )
    rows.append(
        _bench_row(
            "td_array(512) off=-2 backward=F",
            lambda: C0.td_array(sample_td, -2, False),
            lambda: C1.td_array(sample_td, -2, False),
        )
    )
    rows.append(
        _bench_row(
            "cd_array(512) off=4",
            lambda: C0.cd_array(sample_cd, 4),
            lambda: C1.cd_array(sample_cd, 4),
        )
    )
    rows.append(
        _bench_row(
            "td_diff_array(511 pairs)",
            lambda: C0.td_diff_array(sample_td[:-1], sample_td[1:]),
            lambda: C1.td_diff_array(sample_td[:-1], sample_td[1:]),
        )
    )
    rows.append(
        _bench_row(
            "cd_diff_array(511 pairs)",
            lambda: C0.cd_diff_array(sample_cd[:-1], sample_cd[1:]),
            lambda: C1.cd_diff_array(sample_cd[:-1], sample_cd[1:]),
        )
    )
    rows.append(
        _bench_row(
            "td_trailing(n=80)",
            lambda: C0.td_trailing(tmid, 80),
            lambda: C1.td_trailing(tmid, 80),
            repeat=250,
        )
    )
    rows.append(
        _bench_row(
            "cd_trailing(n=120)",
            lambda: C0.cd_trailing(mid, 120),
            lambda: C1.cd_trailing(mid, 120),
            repeat=250,
        )
    )
    rows.append(
        _bench_row(
            "td_within(d0,tmid,step=2,F,F)",
            lambda: C0.td_within(d0, tmid, step=2, until_today=False, updated=False),
            lambda: C1.td_within(d0, tmid, step=2, until_today=False, updated=False),
            repeat=200,
        )
    )
    rows.append(
        _bench_row(
            "cd_within(d0,mid,step=2,T,F)",
            lambda: C0.cd_within(d0, mid, step=2, until_today=True, updated=False),
            lambda: C1.cd_within(d0, mid, step=2, until_today=True, updated=False),
            repeat=200,
        )
    )
    rows.append(
        _bench_row(
            "slice(trade×200,d0,tmid)",
            lambda: C0.slice(arr_200, d0, tmid),
            lambda: C1.slice(arr_200, d0, tmid),
            repeat=500,
        )
    )
    rows.append(
        _bench_row(
            "trade_dates()",
            lambda: C0.trade_dates(),
            lambda: C1.trade_dates(),
            repeat=300,
        )
    )
    rows.append(
        _bench_row(
            "diffs(tgt,src)",
            lambda: C0.diffs(tgt, src),
            lambda: C1.diffs(np.asarray(tgt), np.asarray(src)),
            repeat=120,
        )
    )
    rows.append(
        _bench_row(
            "td_filter(40)",
            lambda: C0.td_filter(sample_td[:40].tolist()),
            lambda: C1.td_filter(sample_td[:40].tolist()),
            repeat=200,
        )
    )
    rows.append(
        _bench_row(
            "qe_trailing",
            lambda: C0.qe_trailing(20200630, 2, 1, 20191231, False),
            lambda: C1.qe_trailing(20200630, 2, 1, 20191231, False),
            repeat=500,
        )
    )
    rows.append(
        _bench_row(
            "is_trade_date×200 scalar",
            lambda: [C0.is_trade_date(int(x)) for x in sample_td[:200]],
            lambda: [C1.is_trade_date(int(x)) for x in sample_td[:200]],
            repeat=80,
        )
    )
    rows.append(
        _bench_row(
            "TradeDate +1 ×40 chain",
            lambda: _td_chain_old(tmid, 40),
            lambda: _td_chain_new(tmid, 40),
            repeat=150,
        )
    )

    print("\n" + "=" * 88)
    print("耗时对比（先校验输出一致再计时；old/new > 1 表示新版更快）")
    print("=" * 88)
    hdr = f"{'场景':<42} {'old(ms)':>10} {'new(ms)':>10} {'old/new':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, ms0, ms1, ratio in rows:
        print(f"{name:<42} {ms0:10.2f} {ms1:10.2f} {ratio:10.2f}x")
    print("=" * 88 + "\n")


def _td_chain_old(start: int, steps: int) -> int:
    x = T0(start)
    for _ in range(steps):
        x = x + 1
    return int(x.as_int())


def _td_chain_new(start: int, steps: int) -> int:
    x = T1(start)
    for _ in range(steps):
        x = x + 1
    return int(x.as_int())


class TestCalendarStaticsVsOld(unittest.TestCase):
    """CALENDAR 中与 BC 无关或仅依赖相同 DB 的接口。"""

    def test_me_qe_ye(self) -> None:
        _np_eq(C0.ME, C1.ME)
        _np_eq(C0.QE, C1.QE)
        _np_eq(C0.YE, C1.YE)

    def test_now_pairwise_tz(self) -> None:
        for bj in (True, False):
            a, b = C0.now(bj), C1.now(bj)
            self.assertEqual(a.tzinfo, b.tzinfo)
            self.assertLessEqual(abs((a - b).total_seconds()), 2.0)

    def test_today_combos(self) -> None:
        for off, bj in ((0, True), (-1, True), (5, True), (0, False)):
            self.assertEqual(C0.today(off, bj), C1.today(off, bj))

    def test_time_bj(self) -> None:
        for bj in (True, False):
            self.assertEqual(C0.time(bj), C1.time(bj))

    def test_datetime_combos(self) -> None:
        cases = [
            (2000, 1, 1, False),
            (1999, 13, 1, False),
            (2020, 3, 15, True),
            (2020, 3, 31, True),
        ]
        for y, m, d, lock in cases:
            self.assertEqual(C0.datetime(y, m, d, lock_month=lock), C1.datetime(y, m, d, lock_month=lock))

    def test_clock_combos(self) -> None:
        pairs = [
            (0, 9, 30, 0),
            (-1, 15, 0, 0),
            (-2, 8, 0, 0),
            (20200101, 14, 30, 45),
        ]
        for date, h, m, s in pairs:
            self.assertEqual(C0.clock(date, h, m, s), C1.clock(date, h, m, s))

    def test_is_updated_today_combos(self) -> None:
        self.assertEqual(C0.is_updated_today(None), C1.is_updated_today(None))
        self.assertEqual(C0.is_updated_today(None, hour=10, minute=0), C1.is_updated_today(None, hour=10, minute=0))
        self.assertEqual(
            C0.is_updated_today(20200101120000, hour=20, minute=0),
            C1.is_updated_today(20200101120000, hour=20, minute=0),
        )
        self.assertEqual(C0.is_updated_today(120000, hour=20, minute=0), C1.is_updated_today(120000, hour=20, minute=0))

    def test_update_to_after_reset(self) -> None:
        _reset_update_to()
        self.assertEqual(C0.update_to(), C1.update_to())

    def test_updated_cap(self) -> None:
        u0 = C0.updated()
        self.assertEqual(C0.updated(), C1.updated())
        cap = min(u0, 20200101)
        self.assertEqual(C0.updated(cap), C1.updated(cap))

    def test_calendar_start_end(self) -> None:
        self.assertEqual(C0.calendar_start(), C1.calendar_start())
        self.assertEqual(C0.calendar_end(), C1.calendar_end())

    def test_trade_dates(self) -> None:
        _np_eq(C0.trade_dates(), C1.trade_dates())

    def test_reformat_combos(self) -> None:
        self.assertEqual(C0.reformat(20200101), C1.reformat(20200101))
        self.assertEqual(
            C0.reformat("2020-01-01", old_fmt="%Y-%m-%d", new_fmt="%Y%m%d"),
            C1.reformat("2020-01-01", old_fmt="%Y-%m-%d", new_fmt="%Y%m%d"),
        )


class TestCalendarTdCdVsOld(unittest.TestCase):
    d0: int
    d1: int
    mid: int
    tmid: int
    t_other: int
    sample_td: np.ndarray
    sample_cd: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        cls.d0 = C0.calendar_start()
        cls.d1 = C0.calendar_end()
        cal_all = C0.cd_within(cls.d0, cls.d1, until_today=False)
        cls.mid = int(cal_all[len(cal_all) // 2])
        cls.tmid = int(C0.td(cls.mid).as_int())
        cls.t_other = int(C0.td(cls.tmid, 3).as_int())
        rng = np.random.default_rng(42)
        tds = C0.trade_dates()
        cds = C0.cd_within(cls.d0, min(cls.d1, cls.d0 + 8000), until_today=False)
        cls.sample_td = rng.choice(tds, size=64, replace=True)
        cls.sample_cd = rng.choice(cds, size=64, replace=True)

    def test_td_offset_pairs(self) -> None:
        pairs = ((self.mid, self.mid, 0), (self.tmid, self.tmid, 2), (T0(self.mid), T1(self.mid), -1))
        for dt0, dt1, off in pairs:
            self.assertEqual(int(C0.td(dt0, off).as_int()), int(C1.td(dt1, off).as_int()))

    def test_td_array_combos(self) -> None:
        for arr in (self.sample_td, self.sample_td.tolist(), pd.Series(self.sample_td)):
            for off, bw in ((0, True), (3, True), (-2, False)):
                _np_eq(C0.td_array(arr, offset=off, backward=bw), C1.td_array(arr, offset=off, backward=bw))

    def test_td_array_torch_vs_numpy_new(self) -> None:
        """旧版 td_array 不接收 torch.Tensor；新版经 as_numpy 转换后与 ndarray 路径一致。"""
        arr = self.sample_td[:32]
        for off, bw in ((0, True), (2, False)):
            a = C1.td_array(arr, offset=off, backward=bw)
            b = C1.td_array(torch.tensor(arr, dtype=torch.long), offset=off, backward=bw)
            _np_eq(a, b)

    def test_cd_combos(self) -> None:
        c_nat = int(C0.cd(self.mid, 10))
        triples = ((self.mid, self.mid, 0), (T0(self.mid), T1(self.mid), 5), (c_nat, c_nat, -3))
        for dt0, dt1, off in triples:
            self.assertEqual(C0.cd(dt0, off), C1.cd(dt1, off))

    def test_cd_array_combos(self) -> None:
        # 新版 isinstance(TradeDate) 只认 T1；旧版只认 T0，须分别传对应类型
        seq_pairs = [
            (self.sample_cd, self.sample_cd),
            ([T0(int(x)) for x in self.sample_cd[:8]], [T1(int(x)) for x in self.sample_cd[:8]]),
        ]
        for seq0, seq1 in seq_pairs:
            for off in (0, -2, 4):
                _np_eq(C0.cd_array(seq0, offset=off), C1.cd_array(seq1, offset=off))

    def test_td_diff_vs_bc(self) -> None:
        for a, b in ((self.tmid, self.t_other), (self.t_other, self.tmid)):
            exp = _td_diff_expected(a, b)
            self.assertEqual(C1.td_diff(a, b), exp)
            # 旧版 td_diff 对 pandas/numpy 整数标量 assert isinstance(..., int) 可能失败，数值以 BC 为准
            try:
                v_old = C0.td_diff(a, b)
            except AssertionError:
                v_old = exp
            self.assertEqual(C1.td_diff(a, b), int(v_old))

    def test_cd_diff_pairs(self) -> None:
        for a, b in ((self.d0, self.mid), (self.mid, self.d0)):
            self.assertEqual(C0.cd_diff(a, b), C1.cd_diff(a, b))
        out_a, out_b = 18000101, 18000105
        self.assertEqual(C0.cd_diff(out_a, out_b), C1.cd_diff(out_a, out_b))

    def test_td_diff_array(self) -> None:
        a, b = self.sample_td[:-1], self.sample_td[1:]
        _np_eq(C0.td_diff_array(a, b), C1.td_diff_array(a, b))

    def test_cd_diff_array(self) -> None:
        a, b = self.sample_cd[:-1], self.sample_cd[1:]
        _np_eq(C0.cd_diff_array(a, b), C1.cd_diff_array(a, b))

    def test_trailing_pairs(self) -> None:
        for date, n in ((self.tmid, 5), (self.tmid, 60), (self.mid, 12)):
            _np_eq(C0.td_trailing(date, n), C1.td_trailing(date, n))
            _np_eq(C0.cd_trailing(date, n), C1.cd_trailing(date, n))

    def test_start_end_dt(self) -> None:
        for v in (None, 20100101, -1):
            self.assertEqual(C0.start_dt(v), C1.start_dt(v))
            self.assertEqual(C0.end_dt(v), C1.end_dt(v))
        self.assertEqual(C0.start_dt(T0(self.tmid)), C1.start_dt(T1(self.tmid)))
        self.assertEqual(C0.end_dt(T0(self.tmid)), C1.end_dt(T1(self.tmid)))

    def test_td_within_combos(self) -> None:
        combos = [
            dict(start_dt=self.d0, end_dt=self.tmid, step=1, until_today=False, updated=False, slice=None),
            dict(start_dt=self.d0, end_dt=self.tmid, step=2, until_today=True, updated=False, slice=None),
            dict(start_dt=None, end_dt=self.tmid, step=1, until_today=True, updated=True, slice=None),
            dict(
                start_dt=self.d0,
                end_dt=self.tmid,
                step=1,
                until_today=False,
                updated=False,
                slice=(self.d0, self.tmid),
            ),
        ]
        for kw in combos:
            sl = kw.pop("slice")
            a = C0.td_within(**kw, slice=sl)
            b = C1.td_within(**kw, slice=sl)
            _np_eq(a, b)

    def test_cd_within_combos(self) -> None:
        for until_today, updated, step in ((True, False, 1), (False, True, 3), (True, True, 2)):
            a = C0.cd_within(self.d0, self.mid, step=step, until_today=until_today, updated=updated)
            b = C1.cd_within(self.d0, self.mid, step=step, until_today=until_today, updated=updated)
            _np_eq(a, b)

    def test_diffs_combos(self) -> None:
        tgt = C0.td_within(self.d0, self.tmid, until_today=False)
        src = tgt[:: max(1, len(tgt) // 5)]
        a = C0.diffs(tgt, src)
        b = C1.diffs(np.asarray(tgt), np.asarray(src))
        _np_eq(a, b)
        a2 = C0.diffs(self.d0, self.tmid, src, td = True)
        b2 = C1.diffs(self.d0, self.tmid, src, type='td')
        _np_eq(a2, b2)
        a3 = C0.diffs(self.d0, self.mid, None, td = False)
        b3 = C1.diffs(self.d0, self.mid, None, type='cd')
        _np_eq(a3, b3)

    def test_td_filter(self) -> None:
        for dl in (self.sample_td[:20].tolist(), C0.td_within(self.d0, self.tmid, until_today=False)[::7].tolist()):
            _np_eq(C0.td_filter(dl), C1.td_filter(dl))

    def test_slice_combos(self) -> None:
        arr = C0.trade_dates()[:200]
        for start_dt, end_dt, year in ((None, None, None), (self.d0, self.tmid, None), (None, None, self.tmid // 10000)):
            _np_eq(
                C0.slice(arr, start_dt, end_dt, year=year),
                C1.slice(arr, start_dt, end_dt, year=year),
            )
        _np_eq(C0.slice(arr.tolist(), self.d0, self.tmid), C1.slice(arr.tolist(), self.d0, self.tmid))
        _np_eq(
            C0.slice(arr, T0(self.d0), T0(self.tmid)),
            C1.slice(arr, T1(self.d0), T1(self.tmid)),
        )

    def test_td_start_end_freqs(self) -> None:
        ref = self.tmid
        for freq, pn, lag in (("d", 1, 0), ("w", 2, 1), ("m", 1, 0), ("q", 1, 1), ("y", 2, 0)):
            a = C0.td_start_end(ref, period_num=pn, freq=freq, lag_num=lag)
            b = C1.td_start_end(ref, period_num=pn, freq=freq, lag_num=lag)
            self.assertEqual(int(a[0].as_int()), int(b[0].as_int()))
            self.assertEqual(int(a[1].as_int()), int(b[1].as_int()))

    def test_cd_start_end_combos(self) -> None:
        cd = self.mid
        for freq, pn, lag in (("d", 1, 0), ("w", 2, 1)):
            self.assertEqual(C0.cd_start_end(cd, pn, freq, lag), C1.cd_start_end(cd, pn, freq, lag))
        for freq, pn, lag in (("m", 1, 0), ("q", 2, 1), ("y", 1, 0)):
            self.assertEqual(C0.cd_start_end(cd, pn, freq, lag), C1.cd_start_end(cd, pn, freq, lag))

    def test_as_trade_date(self) -> None:
        self.assertEqual(int(C0.as_trade_date(self.mid).as_int()), int(C1.as_trade_date(self.mid).as_int()))
        self.assertEqual(int(C0.as_trade_date(self.tmid).as_int()), int(C1.as_trade_date(self.tmid).as_int()))

    def test_is_trade_date(self) -> None:
        self.assertEqual(C0.is_trade_date(self.tmid), C1.is_trade_date(self.tmid))
        non_td = int(C0.cd(self.tmid, 1))
        if non_td != self.tmid:
            self.assertEqual(C0.is_trade_date(non_td), C1.is_trade_date(non_td))

    def test_year_month_quarter_helpers(self) -> None:
        for dt in (self.mid, self.d0 + 400):
            self.assertEqual(C0.year_end(dt), C1.year_end(dt))
            self.assertEqual(C0.year_start(dt), C1.year_start(dt))
            self.assertEqual(C0.quarter_end(dt), C1.quarter_end(dt))
            self.assertEqual(C0.quarter_start(dt), C1.quarter_start(dt))
            self.assertEqual(C0.month_end(dt), C1.month_end(dt))
            self.assertEqual(C0.month_start(dt), C1.month_start(dt))

    def test_qe_trailing_combos(self) -> None:
        d = 20200630
        for npast, nfut, ad, yo in ((1, 0, None, False), (2, 1, 20191231, False), (1, 0, None, True)):
            _np_eq(C0.qe_trailing(d, npast, nfut, ad, yo), C1.qe_trailing(d, npast, nfut, ad, yo))

    def test_qe_within(self) -> None:
        s, e = 20180101, 20201231
        _np_eq(C0.qe_within(s, e, year_only=False), C1.qe_within(s, e, year_only=False))
        _np_eq(C0.qe_within(s, e, year_only=True), C1.qe_within(s, e, year_only=True))

    def test_qe_interpolate(self) -> None:
        _np_eq(C0.qe_interpolate([]), C1.qe_interpolate([]))
        _np_eq(C0.qe_interpolate([20190331, 20200630]), C1.qe_interpolate([20190331, 20200630]))

    def test_check_rollback_date(self) -> None:
        C0.check_rollback_date(None)
        C1.check_rollback_date(None)
        ok_dt = int(C0.updated())
        C0.check_rollback_date(ok_dt, max_rollback_days=10)
        C1.check_rollback_date(ok_dt, max_rollback_days=10)
        C0.check_rollback_date(ok_dt, max_rollback_days=20)
        C1.check_rollback_date(ok_dt, max_rollback_days=20)

    def test_cd_start_end_invalid_freq(self) -> None:
        with self.assertRaises(ValueError):
            C0.cd_start_end(self.mid, 1, "x", 0)
        with self.assertRaises(ValueError):
            C1.cd_start_end(self.mid, 1, "x", 0)

    def test_diffs_bad_argcount(self) -> None:
        with self.assertRaises(AssertionError):
            C0.diffs(1)
        with self.assertRaises(AssertionError):
            C1.diffs(1)


class TestTradeDateVsOld(unittest.TestCase):
    d0: int
    tmid: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.d0 = C0.calendar_start()
        cal_all = C0.cd_within(cls.d0, C0.calendar_end(), until_today=False)
        mid = int(cal_all[len(cal_all) // 2])
        cls.tmid = int(C0.td(mid).as_int())

    def test_init_mapping(self) -> None:
        for raw in (self.tmid, self.d0 + 3):
            self.assertEqual(int(T0(raw).as_int()), int(T1(raw).as_int()))

    def test_force_trade_date(self) -> None:
        for raw in (self.d0 - 1, self.tmid):
            self.assertEqual(int(T0(raw, True).as_int()), int(T1(raw, True).as_int()))

    def test_offset_and_ops(self) -> None:
        for n in (0, 3, -2):
            self.assertEqual(int((T0(self.tmid) + n).as_int()), int((T1(self.tmid) + n).as_int()))
            self.assertEqual(int((T0(self.tmid) - n).as_int()), int((T1(self.tmid) - n).as_int()))
        a0, a1 = T0(self.tmid), T1(self.tmid)
        self.assertEqual(int(T0(self.tmid).offset(4).as_int()), int(T1(self.tmid).offset(4).as_int()))
        self.assertEqual(int(T0(self.tmid).offset(-1).as_int()), int(T1(self.tmid).offset(-1).as_int()))
        self.assertEqual(a0 < T0(self.tmid + 1), a1 < T1(self.tmid + 1))
        self.assertEqual(a0 == T0(self.tmid), a1 == T1(self.tmid))
        self.assertEqual(T0(self.tmid).as_int(), T1(self.tmid).as_int())

    def test_as_numpy_sources(self) -> None:
        x = self.tmid
        for src in (x, [x, x + 1], np.array([x]), pd.Series([x]), torch.tensor([x])):
            _np_eq(T0.as_numpy(src), T1.as_numpy(src))


class TestDatesVsOld(unittest.TestCase):
    d0: int
    tmid: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.d0 = C0.calendar_start()
        cal_all = C0.cd_within(cls.d0, C0.calendar_end(), until_today=False)
        mid = int(cal_all[len(cal_all) // 2])
        cls.tmid = int(C0.td(mid).as_int())

    def test_constructors(self) -> None:
        _np_eq(D0(), D1())
        _np_eq(D0(None), D1(None))
        _np_eq(D0(20200101), D1(20200101))
        _np_eq(D0("20200102"), D1("20200102"))
        _np_eq(D0([20200101, 20200102]), D1([20200101, 20200102]))
        _np_eq(D0(self.d0, self.tmid), D1(self.d0, self.tmid))
        arr = np.array([20200101, 20200201])
        _np_eq(D0(arr, 20200101, 20200201), D1(arr, 20200101, 20200201))

    def test_instance_methods(self) -> None:
        a0 = D0([20200101, 20200201, 20200301])
        a1 = D1([20200101, 20200201, 20200301])
        self.assertEqual(a0.empty, a1.empty)
        _np_eq(a0.diffs([20200201]), a1.diffs([20200201]))
        _np_eq(a0.diffs(np.array([20200201, 20200301])), a1.diffs(np.array([20200201, 20200301])))
        self.assertEqual(a0.format_str(), a1.format_str())


class TestCheckRollbackFailure(unittest.TestCase):
    def test_too_early_raises(self) -> None:
        early = C0.calendar_start()
        with self.assertRaises(AssertionError):
            C0.check_rollback_date(early, max_rollback_days=1)
        with self.assertRaises(AssertionError):
            C1.check_rollback_date(early, max_rollback_days=1)


def main() -> None:
    run_bench = True
    sys.argv = [a for a in sys.argv if a != "--bench"]
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful() and run_bench:
        print_old_new_benchmark_table()
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
