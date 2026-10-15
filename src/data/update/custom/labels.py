"""
Forward return label updater for model training.

Computes 5/10/20-day raw return (``rtn_lag0/1_N``) and risk-model residual return
(``res_lag0/1_N``) labels.  Labels with ``lag1=True`` include a 1-day lag to
avoid execution-day look-ahead bias.

Stored in the ``labels_ts`` database under keys like ``ret5``, ``ret10_lag``, etc.
Fixed-next-day open/VWAP returns use ``retN_open`` / ``retN_vwap`` and fields
``rtn_N_open`` / ``rtn_N_vwap``: T+1 to T+1+N trading days, never T to T+N.
"""
from __future__ import annotations

import pandas as pd
from typing import Literal , TypeAlias

from src.proj import CALENDAR , DB , Base , Dates
from src.data.loader import TRADE , RISK

from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['ClassicLabelsUpdater']

PriceType : TypeAlias = Literal['open' , 'vwap' , 'close'] | str

class ClassicLabelsUpdater(BasicCustomUpdater):
    """
    Registered updater for forward return labels.

    Computes labels for all combinations of ``DAYS × LAGS`` and stores them
    incrementally in ``labels_ts``.
    """
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK)
    START_DATE = 20050101
    DB_SRC = 'labels_ts'

    LABEL_TYPES : tuple[tuple[int,bool],...] = (
        (5 , False) , 
        (10 , False) , 
        (20 , False) , 
        (5 , True) , 
        (10 , True) , 
        (20 , True) , 
        (3 , True) , 
    )
    # Open/VWAP always start at T+1; the database and feature names omit lag1.
    PRICE_LABEL_TYPES = tuple((days, price) for price in ('open', 'vwap') for days in (5, 10, 20))

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """Update classic returns plus fixed-next-day open/VWAP targets."""
        start = max(start or cls.START_DATE , cls.START_DATE)
        end = end or CALENDAR.updated()
        flags = Base.UpdateFlagList()
        jobs = [(days, lag1, 'close') for days, lag1 in cls.LABEL_TYPES]
        jobs += [(days, True, price) for days, price in cls.PRICE_LABEL_TYPES]
        for days , lag1 , price_type in jobs:
            label_name = f'ret{days}' + (('_lag' if lag1 else '') if price_type == 'close' else f'_{price_type}')
            sub_start = CALENDAR.td(start , - days - lag1 + 1).as_int()
            sub_end = CALENDAR.td(CALENDAR.updated() , - days - lag1).as_int()
            stored_dates = Dates() if overwrite else DB.dates(cls.DB_SRC , label_name)
            target_dates = Dates(sub_start , sub_end).slice(cls.START_DATE , end).diff(stored_dates)

            if target_dates.empty:
                cls.logger.skipping(f'{cls.DB_SRC}/{label_name} is up to date' , idt = 1 , vb = 1)
                flags += Base.UpdateFlag.SKIPPED
                continue

            for date in target_dates:
                cls.update_one(date , days , lag1 , label_name, price_type=price_type)

            cls.logger.success(f'Update {cls.DB_SRC}/{label_name} at {Dates(target_dates)}' , idt = 1 , vb = 1)
            flags += Base.UpdateFlag.SUCCESS
        return flags.summarize()

    @classmethod
    def update_one(cls , date : int , days : int , lag1 : bool , label_name : str, *, price_type: PriceType = 'close'):
        """Compute and save labels for a single ``date``."""
        if price_type != 'close' and not lag1:
            raise ValueError('Open/VWAP labels must start on the next trading day')
        label = calc_classic_labels(date, days, lag1) if price_type == 'close' else calc_price_labels(date, days, price_type)
        DB.save(label , cls.DB_SRC , label_name , date , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)


def calc_price_labels(date: int, days: int, price_type: Literal['open', 'vwap'] | str) -> pd.DataFrame | None:
    """Adjusted price return from T+1 to T+1+days, with no extra lag option.

    Only raw returns are stored. Industry/size-neutralised std labels are
    produced by preprocessing; close-based risk residuals are not reused.
    """
    if price_type not in ('open', 'vwap') or days <= 0:
        raise ValueError(f'Invalid next-day price label: {price_type}, {days}')
    d0 = CALENDAR.td(date, 1).as_int()
    d1 = CALENDAR.td(d0, days).as_int()
    label = get_period_ret(d0, d1, price_type)
    if label is None or label.empty:
        return None
    return label.rename(columns={'ret': f'rtn_{days}_{price_type}'}).reset_index()

def get_period_ret(d0 : int , d1 : int , price_type : PriceType = 'close') -> pd.DataFrame | None:
    """
    Period return from ``d0`` to ``d1``.

    Prefers halt-filled ``trade_ts/adjprice`` via ``TRADE.period_return``;
    falls back to ``day * adjfactor`` when adjprice is unavailable.
    """
    return TRADE.period_return(d0 , d1 , price_type)

def calc_classic_labels(
    date : int , days : int , lag1 : bool
) -> pd.DataFrame | None:
    """
    Compute forward return labels for a single date.

    Parameters
    ----------
    date : int
        The label date (yyyyMMdd).  The forward return period starts from
        ``date + lag1`` trading days and ends at ``date + lag1 + days``.
    days : int
        Holding period in trading days (5, 10, or 20).
    lag1 : bool
        If True, adds a 1-day lag between the label date and the start of the
        return period (avoids execution-day look-ahead).

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ``secid``, ``rtn_lag{lag1}_{days}``,
        ``res_lag{lag1}_{days}``.  Returns None if data is unavailable.
    """
    assert days >= 5 or lag1, f'for short term labels ({days} days) , lag1 must be True'
    
    d0 = (CALENDAR.td(date) + lag1).as_int()
    d1 = CALENDAR.td(d0 , days).as_int()

    if days >= 5:
        label_ret = get_period_ret(d0 , d1 , 'close')
        if label_ret is None: 
            return
        label_ret = label_ret.rename(columns={'ret':f'rtn_lag{int(lag1)}_{days}'})
        res1 = RISK.get_res(d1)
        if res1.empty: 
            return
        label_res = RISK.get_exret(d0 , d1).sum().rename(f'res_lag{int(lag1)}_{days}')
        label = pd.merge(label_ret , label_res , on = 'secid').reset_index()
    else:
        label_close = get_period_ret(d0 , d1 , 'close')
        label_open = get_period_ret(d0 , d1 , 'open')
        label_vwap = get_period_ret(d0 , d1 , 'vwap')
        if label_close is None or label_open is None or label_vwap is None: 
            return
        label = label_close.rename(columns={'ret':f'rtn_lag{int(lag1)}_{days}'}).\
            merge(label_open.rename(columns={'ret':f'rtn_lag{int(lag1)}_{days}_open'}) , on = 'secid').\
            merge(label_vwap.rename(columns={'ret':f'rtn_lag{int(lag1)}_{days}_vwap'}) , on = 'secid').reset_index()
    return label
