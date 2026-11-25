#!/usr/bin/env python
# coding: utf-8
# date: 2026-11-25
# description: Reproduce the legacy liquidity factor read from patch 0.7.11/002
# content: 只读检查逐个因子的旧版批量读取和整表标准化，报告失败列、样例和源文件。
# email: False
# mode: shell
# parameters:
#   start:
#       type: int
#       default: 20260413
#       desc: First trading date to inspect
#   end:
#       type: int
#       default: 20261124
#       desc: Last trading date to inspect

from __future__ import annotations

from src.proj import CALENDAR, DB
from src.proj.util.script import ScriptTool
from src.res.factor.calculator import FactorCalculator
from src.res.factor.util.classes.stock_factor import StockFactor


def check(start, end):
    if start > end:
        raise ValueError('start must be on or before end')
    dates = CALENDAR.range(start, end, 'td')
    print(f'Legacy liquidity read: {start}..{end}, {len(dates)} trading dates', flush=True)
    failed, passed, empty = [], 0, 0
    for calc in FactorCalculator.iter(category0='behavior', category1='liquidity'):
        name = calc.factor_name
        try:
            raw = DB.loads(calc.db_src, calc.db_key, dates, override_existing_key=True,
                           closest=False, vb_level='max')
        except Exception as exc:
            print(f'FAIL {name} DB.loads: {type(exc).__name__}: {exc}', flush=True)
            failed.append(name)
            continue
        if raw.empty:
            print(f'EMPTY {name}')
            empty += 1
            continue
        try:
            # Before c0b31734 all returned columns went into the same transform.
            StockFactor.normalize_df(raw, fill_method='drop')
        except Exception as exc:
            failed.append(name)
            print(f'FAIL {name} normalize: {type(exc).__name__}: {exc}', flush=True)
            print(f'  rows={len(raw)} columns={raw.dtypes.astype(str).to_dict()}')
            for col in raw.columns.difference(['secid', 'date']):
                series = raw[col]
                if series.dtype.kind in 'iufb':
                    continue
                text_rows = series.map(lambda value: isinstance(value, str))
                if text_rows.any():
                    row = raw.loc[text_rows, ['date', 'secid', col]].iloc[0]
                    day = int(row['date'])
                    print(f'  column={col!r} date={day} secid={row["secid"]} value={row[col]!r} '
                          f'text_rows={int(text_rows.sum())} file={calc.target_path(day)}')
            continue
        print(f'OK {name} rows={len(raw)}')
        passed += 1
    print(f'Summary: ok={passed} empty={empty} failed={len(failed)} factors={failed}', flush=True)
    return 1 if failed else (0 if passed else 2)


@ScriptTool('reproduce_liquidity_read', email=False)
def main(start: int = 20260413, end: int = 20261124):
    return check(start, end)


if __name__ == '__main__':
    task = main()
    raise SystemExit(task.func_return if task.execution_success else 1)
