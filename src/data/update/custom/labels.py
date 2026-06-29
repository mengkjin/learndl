"""
Forward return label updater for model training.

Computes 5/10/20-day raw return (``rtn_lag0/1_N``) and risk-model residual return
(``res_lag0/1_N``) labels.  Labels with ``lag1=True`` include a 1-day lag to
avoid execution-day look-ahead bias.

Stored in the ``labels_ts`` database under keys like ``ret5``, ``ret10_lag``, etc.
"""
from __future__ import annotations

import pandas as pd
from typing import Literal , TypeAlias

from src.proj import CALENDAR , DB , Base , Dates
from src.data.loader import TRADE , RISK

from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['ClassicLabelsUpdater']

PriceType : TypeAlias = Literal['open' , 'vwap' , 'close']

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

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """Update forward return labels for all combinations of ``DAYS × LAGS``."""
        start = max(start or cls.START_DATE , cls.START_DATE)
        end = end or CALENDAR.updated()
        flags = Base.UpdateFlagList()
        for days , lag1 in cls.LABEL_TYPES:
            label_name = f'ret{days}' + ('_lag' if lag1 else '')
            sub_start = CALENDAR.td(start , - days - lag1 + 1).as_int()
            sub_end = CALENDAR.td(CALENDAR.updated() , - days - lag1).as_int()
            stored_dates = Dates() if overwrite else DB.dates(cls.DB_SRC , label_name)
            target_dates = Dates(sub_start , sub_end).slice(cls.START_DATE , end).diff(stored_dates)

            if target_dates.empty:
                cls.logger.skipping(f'{cls.DB_SRC}/{label_name} is up to date' , idt = 1 , vb = 1)
                flags += Base.UpdateFlag.SKIPPED
                continue

            for date in target_dates:
                cls.update_one(date , days , lag1 , label_name)

            cls.logger.success(f'Update {cls.DB_SRC}/{label_name} at {Dates(target_dates)}' , idt = 1 , vb = 1)
            flags += Base.UpdateFlag.SUCCESS
        return flags.summarize()

    @classmethod
    def update_one(cls , date : int , days : int , lag1 : bool , label_name : str):
        """Compute and save labels for a single ``date``."""
        DB.save(calc_classic_labels(date , days , lag1) , cls.DB_SRC , label_name , date , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)

def get_period_ret(d0 : int , d1 : int , price_type : PriceType = 'close') -> pd.DataFrame | None:
    """Get the period return for a single date."""
    q1 = TRADE.get_trd(d1)
    if q1.empty: 
        return
    q1 = q1.rename(columns={'adjfactor':'adj1' , price_type:'p1'})[['secid','adj1','p1']]
    q0 = TRADE.get_trd(d0).rename(columns={'adjfactor':'adj0' , price_type:'p0'})[['secid','adj0','p0']]
    ret = q1.merge(q0 , how = 'left' , on = 'secid').set_index('secid')
    label_ret = (ret['p1'] * ret['adj1'].fillna(1) / ret['p0'] / ret['adj0'].fillna(1) - 1).rename(f'ret').to_frame()
    return label_ret

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
        label = label_close.rename(columns={'ret':f'ret_lag{int(lag1)}_{days}'}).\
            merge(label_open.rename(columns={'ret':f'ret_lag{int(lag1)}_{days}_open'}) , on = 'secid').\
            merge(label_vwap.rename(columns={'ret':f'ret_lag{int(lag1)}_{days}_vwap'}) , on = 'secid').reset_index()
    return label