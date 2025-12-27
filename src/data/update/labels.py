import pandas as pd

import numpy as np
from typing import Literal

from src.proj import CALENDAR , DB , Logger
from src.data.loader import TRADE , RISK

from .basic import BasicUpdater

class ClassicLabelsUpdater(BasicUpdater):
    START_DATE = 20050101
    DB_SRC = 'labels_ts'

    DAYS = [5 , 10 , 20]
    LAGS = [False , True]

    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : int = 1):
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all classic labels is not supported yet for {cls.__name__}')
        for days in cls.DAYS:
            for lag1 in cls.LAGS:
                label_name = f'ret{days}' + ('_lag' if lag1 else '')
                if update_type == 'recalc':
                    stored_dates = np.array([])
                elif update_type == 'update':
                    stored_dates = DB.dates(cls.DB_SRC , label_name)
                elif update_type == 'rollback':
                    rollback_date = CALENDAR.td(cls._rollback_date , - days - lag1 + 1)
                    stored_dates = CALENDAR.slice(DB.dates(cls.DB_SRC , label_name) , 0 , rollback_date - 1)
                else:
                    raise ValueError(f'Invalid update type: {update_type}')
                end_date     = CALENDAR.td(CALENDAR.updated() , - days - lag1)
                update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
                for date in update_dates:
                    cls.update_one(date , days , lag1 , label_name , indent = indent + 1 , vb_level = vb_level + 2)

            Logger.success(f'Update {cls.DB_SRC}/{label_name} at {CALENDAR.dates_str(update_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , days : int , lag1 : bool , label_name : str , indent : int = 2 , vb_level : int = 2):
        DB.save(calc_classic_labels(date , days , lag1) , cls.DB_SRC , label_name , date , indent = indent , vb_level = vb_level)

def calc_classic_labels(date : int , days : int , lag1 : bool) -> pd.DataFrame | None:
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