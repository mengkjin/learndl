"""
Multi-period OHLCV aggregation updater.

Computes 5-day, 10-day, and 20-day OHLCV bars from daily data with proper
adjfactor treatment and stores them in ``trade_ts/{N}day``.

The aggregation uses OHLCV rules: open=first, high=max, low=min, close=last,
amount/volume/turnover=sum, pctchange=product.  VWAP is recomputed from
total amount and volume.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from src.proj import CALENDAR , DB , Base , Dates
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['MultiKlineUpdater']

class MultiKlineUpdater(BasicCustomUpdater):
    """Registered updater for 5/10/20-day aggregated OHLCV bars."""
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE , Base.UpdateType.ROLLBACK)
    START_DATE = 20050101
    DB_SRC = 'trade_ts'

    DAYS = [5 , 10 , 20]

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """Update multi-day OHLCV aggregations for all (n_day, missing_dates) combinations."""
        start = max(start or cls.START_DATE , cls.START_DATE)
        flags = Base.UpdateFlagList()
        for n_day in cls.DAYS:
            label_name = f'{n_day}day'
            sub_end = DB.dates(cls.DB_SRC , 'day').max
            stored_dates = Dates() if overwrite else DB.dates(cls.DB_SRC , label_name)
            target_dates = Dates(start , sub_end).diff(stored_dates)

            if target_dates.empty:
                cls.logger.skipping(f'{cls.DB_SRC}/{label_name} is up to date' , idt = 1 , vb = 1)
                flags += Base.UpdateFlag.SKIPPED
                continue

            for date in target_dates: 
                cls.update_one(date , n_day , label_name)
            cls.logger.success(f'Update {cls.DB_SRC}/{label_name} at {target_dates}' , idt = 1 , vb = 1)
            flags += Base.UpdateFlag.SUCCESS
        return flags.summarize()

    @classmethod
    def update_one(cls , date : int , n_day : int , label_name : str):
        """Compute and save the n-day OHLCV bar for a single ``date``."""
        DB.save(nday_kline(date , n_day) , cls.DB_SRC , label_name , date , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)

def nday_kline(date : int , n_day : int) -> pd.DataFrame:
    """
    Aggregate trailing ``n_day`` daily bars into a single OHLCV row per secid.

    Prices are adjusted by ``adjfactor`` before aggregation.  VWAP is
    recomputed as ``sum(amount) / sum(volume)``, falling back to close
    when volume is zero.  Returns an empty DataFrame if no data is available.
    """
    # read calendar
    assert n_day in [5 , 10 , 20] , f'n_day should be in [5 , 10 , 20]'
    trailing_dates = CALENDAR.trailing(date , n_day , 'td')
    assert trailing_dates[-1] == date , (trailing_dates[-1] , date)

    price_feat  = ['open','close','high','low','vwap']
    volume_feat = ['amount','volume','turn_tt','turn_fl','turn_fr']

    datas = [DB.load('trade_ts' , 'day' , d , key_column='date') for d in trailing_dates]
    datas = [d for d in datas if not d.empty]
    if not datas: 
        return pd.DataFrame()
    with np.errstate(invalid='ignore' , divide = 'ignore'):
        data = pd.concat(datas , axis = 0).sort_values(['secid','date'])
        data.loc[:,'adjfactor'] = data.loc[:,'adjfactor'].ffill().fillna(1)
        data.loc[:,price_feat] = data.loc[:,price_feat] * data.loc[:,'adjfactor'].to_numpy(float)[:,None]
        data['pctchange'] = data['pctchange'] / 100 + 1
        data['vwap'] = data['vwap'] * data['volume']
        agg_dict = {'open':'first','high':'max','low':'min','close':'last','pctchange':'prod','vwap':'sum',**{k:'sum' for k in volume_feat},}
        df : pd.DataFrame | Any = data.groupby('secid').agg(agg_dict)
        df['pctchange'] = (df['pctchange'] - 1) * 100
        df['vwap'] /= np.where(df['volume'] == 0 , np.nan , df['volume'])
        df['vwap'] = df['vwap'].where(~df['vwap'].isna() , df['close'])
    return df
