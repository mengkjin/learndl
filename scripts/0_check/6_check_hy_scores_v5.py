# coding: utf-8
# author: jinmeng
# date: 2026-09-24
# description: Check hy_scores_v5 null coverage
# content: |
#   检查华源 sellside 因子（默认 hy_scores_v5）在指定年份的截面非空情况。
#   逐日统计：行数、非空数、空值率；并标出全缺失日与日历缺口。
#   这正是 FactorStatsUpdater 首次归一化时 No objects to concatenate 的触发条件。
# email: False
# mode: shell
# parameters:
#   factor_name:
#       type: [hy_scores_v5, hy_cmm]
#       desc: affiliate sellside factor to audit
#       required: False
#       default: hy_scores_v5
#   year:
#       type: int
#       desc: calendar year to audit (YYYY)
#       required: False
#       default: 2026
#   start:
#       type: int
#       desc: inclusive start yyyyMMdd; 0 = year0101
#       required: False
#       default: 0
#   end:
#       type: int
#       desc: inclusive end yyyyMMdd; 0 = year1231 clamped to today
#       required: False
#       default: 0
#   min_valid:
#       type: int
#       desc: warn when a date has fewer non-null values than this
#       required: False
#       default: 100
#   save_csv:
#       type: [True, False]
#       desc: write per-date coverage CSV under PATH.runtime
#       required: False
#       default: True

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proj import CALENDAR, PATH, Logger
from src.proj.util.script import ScriptTool
from src.res.factor.calculator.factor_calc import AffiliateFactorCalculator, FactorCalculator


def _optional_date(value: int | str | None) -> int | None:
    if value is None or value == '':
        return None
    text = str(value).strip()
    if text in {'0', 'None'}:
        return None
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f'Expected YYYYMMDD, got {value!r}')
    datetime.strptime(text, '%Y%m%d')
    return int(text)


def _as_bool(value: bool | str | None) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _year_bounds(year: int) -> tuple[int, int]:
    return int(f'{year}0101'), int(f'{year}1231')


def _daily_coverage(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per date: row count and finite non-null factor values."""
    if df.empty:
        return pd.DataFrame(columns=[
            'date', 'n_rows', 'n_valid', 'n_null', 'null_rate', 'valid_rate',
        ])
    work = df.loc[:, ['date', col]].copy()
    work['date'] = work['date'].astype(int)
    # Treat both NA and non-finite values as unusable for normalize/stats.
    values = pd.to_numeric(work[col], errors='coerce')
    work['is_valid'] = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    grouped = work.groupby('date', sort=True)
    out = grouped.agg(n_rows=('is_valid', 'size'), n_valid=('is_valid', 'sum')).reset_index()
    out['n_null'] = out['n_rows'] - out['n_valid']
    out['null_rate'] = (out['n_null'] / out['n_rows']).where(out['n_rows'] > 0, np.nan)
    out['valid_rate'] = (out['n_valid'] / out['n_rows']).where(out['n_rows'] > 0, np.nan)
    return out


def _classify(row: pd.Series, min_valid: int) -> str:
    if int(row['n_rows']) == 0:
        return 'empty_file'
    if int(row['n_valid']) == 0:
        return 'all_null'
    if int(row['n_valid']) < min_valid:
        return 'low_valid'
    return 'ok'


def _print_table(title: str, frame: pd.DataFrame, cols: list[str], limit: int = 20) -> None:
    Logger.stdout(f'--- {title} ---')
    if frame.empty:
        Logger.stdout('(none)')
        return
    show = frame.loc[:, cols].head(limit)
    Logger.stdout(show.to_string(index=False))
    if len(frame) > limit:
        Logger.stdout(f'... {len(frame) - limit} more rows')


@ScriptTool('check_hy_scores_coverage')
def main(
    factor_name: str = 'hy_scores_v5',
    year: int = 2026,
    start: int = 0,
    end: int = 0,
    min_valid: int = 100,
    save_csv: bool = True,
    **kwargs,
):
    """Summarize cross-sectional non-null coverage for a sellside affiliate factor."""
    del kwargs
    year_i = int(year)
    if year_i < 1990 or year_i > 2100:
        raise ValueError(f'year out of range: {year_i}')
    year_start, year_end = _year_bounds(year_i)
    start_i = _optional_date(start) or year_start
    end_i = _optional_date(end) or min(year_end, int(datetime.now().strftime('%Y%m%d')))
    min_valid_i = int(min_valid)
    save_csv_i = _as_bool(save_csv)
    if start_i > end_i or min_valid_i < 0:
        raise ValueError('Require start <= end and min_valid >= 0')

    FactorCalculator.import_definitions()
    calc = FactorCalculator.get(factor_name)
    if not isinstance(calc, AffiliateFactorCalculator) or calc.category1 != 'sellside':
        raise ValueError(f'{factor_name} is not a sellside affiliate factor')

    stored = calc.stored_dates(start=start_i, end=end_i)
    # until_today=False: audit the requested window even if local calendar update lags.
    calendar = CALENDAR.range(start_i, end_i, 'td', until_today=False)
    stored_arr = stored.dates if len(stored) else np.array([], dtype=int)
    missing_dates = np.setdiff1d(calendar, stored_arr) if len(calendar) else np.array([], dtype=int)

    Logger.stdout(
        f'{factor_name} coverage audit: [{start_i}, {end_i}] '
        f'db={calc.load_db_src}/{calc.load_db_key} col={calc.load_db_col}'
    )
    Logger.stdout(
        f'trading days={len(calendar)}, stored files={len(stored)}, '
        f'calendar gaps={len(missing_dates)}'
    )
    if len(calendar) == 0:
        Logger.alert1(
            f'No trading calendar dates in [{start_i}, {end_i}]; '
            'gap checks skipped, stored-file coverage still audited.'
        )

    raw = calc.Loads(stored_arr)
    col = calc.factor_name
    if not raw.empty and col not in raw.columns:
        raise RuntimeError(f'Loaded frame missing factor column {col!r}; got {list(raw.columns)}')

    daily = _daily_coverage(raw, col)
    if len(stored) and not daily.empty:
        # Dates present in DB.dates but absent after Loads (schema/rename issues).
        lost = np.setdiff1d(stored_arr, daily['date'].to_numpy(int))
    else:
        lost = np.array([], dtype=int)
    if len(lost):
        Logger.alert1(f'{len(lost)} stored dates disappeared after Loads: {lost[:10]}')

    if daily.empty:
        daily = pd.DataFrame(columns=[
            'date', 'n_rows', 'n_valid', 'n_null', 'null_rate', 'valid_rate', 'status',
        ])
    else:
        daily['status'] = daily.apply(lambda row: _classify(row, min_valid_i), axis=1)

    status_counts = (
        daily['status'].value_counts().reindex(
            ['ok', 'low_valid', 'all_null', 'empty_file'], fill_value=0
        )
        if not daily.empty else
        pd.Series({'ok': 0, 'low_valid': 0, 'all_null': 0, 'empty_file': 0})
    )

    Logger.stdout('status counts:')
    for name, count in status_counts.items():
        Logger.stdout(f'  {name}: {int(count)}')
    Logger.stdout(f'  calendar_gap: {len(missing_dates)}')

    if not daily.empty:
        Logger.stdout(
            'valid summary: '
            f'min={int(daily["n_valid"].min())}, '
            f'median={float(daily["n_valid"].median()):.0f}, '
            f'mean={float(daily["n_valid"].mean()):.1f}, '
            f'max={int(daily["n_valid"].max())}'
        )
        Logger.stdout(
            'null_rate summary: '
            f'min={float(daily["null_rate"].min()):.4f}, '
            f'median={float(daily["null_rate"].median()):.4f}, '
            f'max={float(daily["null_rate"].max()):.4f}'
        )

    all_null = daily.loc[daily['status'] == 'all_null'].sort_values('date')
    low_valid = daily.loc[daily['status'] == 'low_valid'].sort_values(['n_valid', 'date'])
    worst_null = daily.sort_values(['valid_rate', 'n_valid', 'date']).head(20)

    _print_table(
        'all-null dates (normalize crash trigger)',
        all_null,
        ['date', 'n_rows', 'n_valid', 'n_null', 'null_rate', 'status'],
    )
    _print_table(
        f'low-valid dates (n_valid < {min_valid_i})',
        low_valid,
        ['date', 'n_rows', 'n_valid', 'n_null', 'null_rate', 'status'],
    )
    _print_table(
        'worst coverage dates',
        worst_null,
        ['date', 'n_rows', 'n_valid', 'n_null', 'null_rate', 'status'],
    )
    if len(missing_dates):
        sample = missing_dates[:20]
        Logger.stdout(f'--- calendar gaps (first {len(sample)}) ---')
        Logger.stdout(', '.join(str(int(d)) for d in sample))
        if len(missing_dates) > 20:
            Logger.stdout(f'... {len(missing_dates) - 20} more')

    # Stats updater would feed Factor() only stored dates in the year.
    trigger = int(status_counts.get('all_null', 0)) + int(status_counts.get('empty_file', 0))
    if trigger > 0:
        Logger.alert1(
            f'{trigger} stored dates have zero usable values; '
            'FactorStatsUpdater normalize can hit "No objects to concatenate" on first stats.'
        )
    elif len(stored) == 0:
        Logger.alert1('No stored sellside files in range; stats job would load an empty factor.')
    else:
        Logger.success('Every stored date has at least one finite factor value.')

    if save_csv_i:
        out_dir = PATH.runtime / 'factor_coverage' / datetime.now().strftime('%Y%m%d-%H%M%S')
        out_dir.mkdir(parents=True, exist_ok=True)
        daily_path = out_dir / f'{factor_name}_{year_i}_daily.csv'
        gap_path = out_dir / f'{factor_name}_{year_i}_calendar_gaps.csv'
        summary_path = out_dir / f'{factor_name}_{year_i}_summary.csv'
        daily.to_csv(daily_path, index=False)
        pd.DataFrame({'date': missing_dates.astype(int)}).to_csv(gap_path, index=False)
        summary = pd.DataFrame([{
            'factor_name': factor_name,
            'db_src': calc.load_db_src,
            'db_key': calc.load_db_key,
            'col': calc.load_db_col,
            'start': start_i,
            'end': end_i,
            'trading_days': len(calendar),
            'stored_dates': len(stored),
            'calendar_gaps': len(missing_dates),
            'ok': int(status_counts.get('ok', 0)),
            'low_valid': int(status_counts.get('low_valid', 0)),
            'all_null': int(status_counts.get('all_null', 0)),
            'empty_file': int(status_counts.get('empty_file', 0)),
            'min_valid_threshold': min_valid_i,
        }])
        summary.to_csv(summary_path, index=False)
        Logger.stdout(f'CSV reports: {out_dir}')


if __name__ == '__main__':
    task = main()
    if getattr(task, 'status', 'Success') == 'Error':
        raise SystemExit(1)
