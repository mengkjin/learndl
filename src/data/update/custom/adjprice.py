"""
Adjusted-price panel with halt forward-fill.

For each trading day, store ``open/high/low/close/vwap`` already multiplied by
``adjfactor``.  Names missing from ``trade_ts/day`` (typical A-share halt)
are carried from the previous trading day's ``adjprice`` row. Unknown listing
metadata does not exclude a security; known pre-IPO and delisted names are removed,
so any two-date
period return is simply ``p1 / p0 - 1``.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import os
import fcntl
from uuid import uuid4

from src.proj import CALENDAR , DB , Base , Dates , Save
from src.data.util import INFO
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['AdjPriceUpdater' , 'calc_adjprice']

PRICE_COLS : tuple[str , ...] = ('open' , 'high' , 'low' , 'close' , 'vwap')


class AdjPriceUpdater(BasicCustomUpdater):
    """Write ``trade_ts/adjprice`` after daily quotes (and for historical backfill)."""
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK , Base.UpdateType.RECALC)
    START_DATE = 19970101
    DB_SRC = 'trade_ts'
    DB_KEY = 'adjprice'

    @classmethod
    def proceed_update(
        cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """
        Build adjprice chronologically for ``[start, end]``.

        Must run in date order so halt fill can read the previous day's panel.
        """
        day_dates = DB.dates(cls.DB_SRC , 'day')
        if day_dates.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/day is empty — nothing to build' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        start = int(start) if start is not None else int(day_dates.min)
        sub_end = int(end) if end else int(day_dates.max)
        stored = Dates() if overwrite else DB.dates(cls.DB_SRC , cls.DB_KEY)
        target = Dates(start , sub_end).intersect(day_dates).diff(stored)
        if target.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        for date in target:
            cls.update_one(int(date))
        cls.logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {Dates(target)}' , idt = 1 , vb = 1)
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int) -> None:
        """Compute and save adjprice for a single trading ``date``."""
        path = DB.path(cls.DB_SRC, cls.DB_KEY, int(date))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Daily updates and coverage repairs share a per-date writer lock.
        with path.with_suffix('.repair.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            candidate = calc_adjprice(int(date))
            temporary = path.with_name(f'.{path.stem}.{uuid4().hex}.feather')
            try:
                Save.df(candidate, temporary, vb_level='never')
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


def exclude_inactive(frame: pd.DataFrame, date: int) -> pd.DataFrame:
    """Unknown metadata is not an exclusion; known IPO/delisting bounds are."""
    desc = INFO.get_desc(set_index=False, listed=False)
    # A code may have several listing intervals (e.g. 600018). Exclude it
    # only when every record is inactive, not when any single record is.
    outside = ((desc.list_dt > 0) & (desc.list_dt > int(date))) | \
              ((desc.delist_dt > 0) & (desc.delist_dt <= int(date)))
    active = desc.loc[~outside, 'secid']
    inactive = desc.loc[~desc.secid.isin(active), 'secid']
    return frame.loc[~frame.secid.isin(inactive)].copy()


def covers_day(adj: pd.DataFrame, day: pd.DataFrame, date: int) -> bool:
    """Check valid daily price cells, without requiring unavailable raw prices."""
    if day is None or day.empty or not {'secid', 'adjfactor'}.issubset(day.columns):
        return True
    expected = exclude_inactive(day, date).set_index('secid')
    if expected.empty:
        return True
    if adj is None or adj.empty or 'secid' not in adj.columns or adj.secid.duplicated().any():
        return False
    actual = adj.set_index('secid').reindex(expected.index)
    factor = expected.adjfactor.fillna(1.)
    for column in PRICE_COLS:
        if column not in expected.columns:
            continue
        raw = expected[column] * factor
        valid = np.isfinite(raw) & (raw > 0)
        if valid.any() and (column not in actual.columns or
                           not (np.isfinite(actual.loc[valid, column]) & (actual.loc[valid, column] > 0)).all()):
            return False
    return True



def calc_adjprice(date : int, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Adjusted OHLC/VWAP for the union of daily quotes and previous prices.

    - Traded names: ``price * adjfactor`` from ``trade_ts/day``.
    - Absent from day (halt): carry previous ``adjprice`` row.
    - Exclude only known pre-listing/delisted names; keep unknown metadata.
    - Still-all-NaN rows (e.g. IPO with no trade yet) are dropped.
    """
    date = int(date)
    day = DB.load('trade_ts' , 'day' , date , vb_level = 'never')
    required = {'secid', 'adjfactor', *PRICE_COLS}
    if day is None or day.empty or not required.issubset(day.columns):
        raise ValueError(f'Cannot calculate adjprice at {date}: trade_ts/day is missing, empty or lacks required columns')
    cols = ['secid' , *PRICE_COLS]
    day = exclude_inactive(day, date)
    if day.secid.duplicated().any():
        raise ValueError(f'Cannot calculate adjprice at {date}: duplicate daily securities')

    day = day.copy()
    adj = day['adjfactor'].fillna(1.0).to_numpy(float)
    traded = pd.DataFrame({'secid': day['secid'].to_numpy()})
    for c in PRICE_COLS:
        traded[c] = day[c].to_numpy(float) * adj
    traded = traded.set_index('secid')

    prev = CALENDAR.td(date, -1).as_int()
    if previous is None:
        previous = (DB.load('trade_ts', 'adjprice', prev, vb_level='never')
                    if prev >= CALENDAR.calendar_start() else pd.DataFrame())
    if previous is not None and not previous.empty:
        previous = exclude_inactive(previous, date)
        if previous.secid.duplicated().any():
            raise ValueError(f'Cannot calculate adjprice at {date}: duplicate previous securities')
        prior = previous.set_index('secid').reindex(columns=list(PRICE_COLS))
        result = traded.combine_first(prior)
    else:
        result = traded

    result = result.dropna(how = 'all' , subset = list(PRICE_COLS))
    return result.reset_index().loc[: , cols]
