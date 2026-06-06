"""
Forward return label updater for model training.

Computes 5/10/20-day raw return (``rtn_lag0/1_N``) and risk-model residual return
(``res_lag0/1_N``) labels.  Labels with ``lag1=True`` include a 1-day lag to
avoid execution-day look-ahead bias.

Stored in the ``labels_ts`` database under keys like ``ret5``, ``ret10_lag``, etc.
"""
from __future__ import annotations
import pandas as pd

import numpy as np
from typing import Literal

from src.proj import CALENDAR , DB , Base
from src.data.loader import TRADE , RISK

from src.data.update.custom.basic import BasicCustomUpdater

class ClassicLabelsUpdater(BasicCustomUpdater):
    """
    Registered updater for forward return labels.

    Computes labels for all combinations of ``DAYS × LAGS`` and stores them
    incrementally in ``labels_ts``.
    """
    START_DATE = 20050101
    DB_SRC = 'labels_ts'

    DAYS = [5 , 10 , 20]
    LAGS = [False , True]

    def update_all(self , update_type : Literal['recalc' , 'update' , 'rollback']):
        """Iterate over all (days, lag) combinations and update any missing dates."""
        if update_type == 'recalc':
            self.logger.warning(f'Recalculate all classic labels is not supported yet for {self.__class__.__name__}')
        for days in self.DAYS:
            for lag1 in self.LAGS:
                label_name = f'ret{days}' + ('_lag' if lag1 else '')
                if update_type == 'recalc':
                    stored_dates = np.array([])
                elif update_type == 'update':
                    stored_dates = DB.dates(self.DB_SRC , label_name)
                elif update_type == 'rollback':
                    rollback_date = CALENDAR.td(self._rollback_date , - days - lag1 + 1)
                    stored_dates = CALENDAR.slice(DB.dates(self.DB_SRC , label_name) , 0 , rollback_date - 1)
                else:
                    raise ValueError(f'Invalid update type: {update_type}')
                end = CALENDAR.td(CALENDAR.updated() , - days - lag1)
                update_dates = CALENDAR.diffs(self.START_DATE , end , stored_dates)
                if len(update_dates) == 0:
                    self.logger.skipping(f'{self.DB_SRC}/{label_name} is up to date' , idt = 1 , vb = 1)
                    continue
                for date in update_dates:
                    self.update_one(date , days , lag1 , label_name)

                self.logger.success(f'Update {self.DB_SRC}/{label_name} at {Base.Dates(update_dates)}' , idt = 1 , vb = 1)

    def update_one(self , date : int , days : int , lag1 : bool , label_name : str):
        """Compute and save labels for a single ``date``."""
        DB.save(calc_classic_labels(date , days , lag1) , self.DB_SRC , label_name , date , indent = self.indent + 2 , vb_level = self.vb_level + 2)

def calc_classic_labels(date : int , days : int , lag1 : bool) -> pd.DataFrame | None:
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
    d0 = CALENDAR.td(date) + lag1
    d1 = CALENDAR.td(d0 , days)

    q1 = TRADE.get_trd(d1)
    res1 = RISK.get_res(d1)
    if q1.empty or res1.empty: 
        return

    q1 = q1.rename(columns={'adjfactor':'adj1' , 'close':'cp1'})[['secid','adj1','cp1']]
    q0 = TRADE.get_trd(d0).rename(columns={'adjfactor':'adj0' , 'close':'cp0'})[['secid','adj0','cp0']]

    ret = q1.merge(q0 , how = 'left' , on = 'secid').set_index('secid')
    label_ret = (ret['cp1'] * ret['adj1'].fillna(1) / ret['cp0'] / ret['adj0'].fillna(1) - 1).rename(f'rtn_lag{int(lag1)}_{days}')
    label_res = RISK.get_exret(d0 , int(d1)).sum().rename(f'res_lag{int(lag1)}_{days}')

    label = pd.merge(label_ret , label_res , on = 'secid').reset_index()
    return label