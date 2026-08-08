"""
Adjusted-price panel with halt forward-fill.

For each trading day, store ``open/high/low/close/vwap`` already multiplied by
``adjfactor``.  Listed names missing from ``trade_ts/day`` (typical A-share halt)
are filled from the previous trading day's ``adjprice`` row so any two-date
period return is simply ``p1 / p0 - 1``.
"""
from __future__ import annotations

import pandas as pd

from src.proj import CALENDAR , DB , Base , Dates
from src.data.util import INFO
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['AdjPriceUpdater' , 'calc_adjprice']

PRICE_COLS : tuple[str , ...] = ('open' , 'high' , 'low' , 'close' , 'vwap')


class AdjPriceUpdater(BasicCustomUpdater):
    """Write ``trade_ts/adjprice`` after daily quotes (and for historical backfill)."""
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK , Base.UpdateType.RECALC)
    START_DATE = 20050101
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
        start = max(int(start or cls.START_DATE) , cls.START_DATE)
        day_dates = DB.dates(cls.DB_SRC , 'day')
        if day_dates.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/day is empty — nothing to build' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

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
        DB.save(
            calc_adjprice(int(date)) ,
            cls.DB_SRC ,
            cls.DB_KEY ,
            int(date) ,
            indent = cls.logger.indent + 2 ,
            vb_level = cls.logger.vb_level + 2 ,
        )


def calc_adjprice(date : int) -> pd.DataFrame:
    """
    Adjusted OHLC/VWAP for all names listed on ``date``.

    - Traded names: ``price * adjfactor`` from ``trade_ts/day``.
    - Listed but absent from day (halt): carry previous ``adjprice`` row.
    - Still-all-NaN rows (e.g. IPO with no trade yet) are dropped.
    """
    date = int(date)
    listed = INFO.get_secid(date)
    cols = ['secid' , *PRICE_COLS]
    if listed.size == 0:
        return pd.DataFrame(columns = cols)

    day = DB.load('trade_ts' , 'day' , date , vb_level = 'never')
    if day is None or day.empty:
        traded = pd.DataFrame(columns = PRICE_COLS)
    else:
        day = day.copy()
        adj = day['adjfactor'].fillna(1.0).to_numpy(float)
        traded = pd.DataFrame({'secid': day['secid'].to_numpy()})
        for c in PRICE_COLS:
            traded[c] = day[c].to_numpy(float) * adj
        traded = traded.set_index('secid')

    univ = pd.Index(listed , name = 'secid')
    result = traded.reindex(univ)

    need_fill = bool(result[list(PRICE_COLS)].isna().to_numpy().any())
    if need_fill:
        prev = CALENDAR.td(date , -1).as_int()
        if prev >= AdjPriceUpdater.START_DATE:
            prev_df = DB.load('trade_ts' , 'adjprice' , prev , vb_level = 'never')
            if prev_df is not None and not prev_df.empty:
                prev_px = prev_df.set_index('secid').reindex(univ).loc[: , list(PRICE_COLS)]
                result = result.fillna(prev_px)

    result = result.dropna(how = 'all' , subset = list(PRICE_COLS))
    return result.reset_index().loc[: , cols]
