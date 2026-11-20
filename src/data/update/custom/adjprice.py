"""Registered adjusted-price update; calculations live in the price service."""
from __future__ import annotations
from src.proj import Base, DB
from src.data.util.prices.adjusted import ensure_adjprices, START_DATE
from .basic import BasicCustomUpdater

__all__ = ['AdjPriceUpdater']

class AdjPriceUpdater(BasicCustomUpdater):
    START_DATE = START_DATE
    UPDATE_ORDER = -100
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE, Base.UpdateType.ROLLBACK, Base.UpdateType.RECALC)
    DB_SRC = 'trade_ts'
    DB_KEY = 'adjprice'

    @classmethod
    def proceed_update(cls, start=None, end=None, overwrite=False, **kwargs):
        dates = DB.dates('trade_ts', 'day')
        selected = [int(date) for date in dates
                    if (start is None or date >= start) and (end is None or date <= end)]
        changed = ensure_adjprices(selected, force=overwrite, reason='custom') if selected else False
        return Base.UpdateFlag.SUCCESS if changed else Base.UpdateFlag.SKIPPED
