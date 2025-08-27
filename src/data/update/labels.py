import pandas as pd

from typing import Optional

from src.basic import PATH , CALENDAR
from src.data.loader import TRADE , RISK

class ClassicLabelsUpdater:
    START_DATE = 20050101
    DB_SRC = 'labels_ts'

    DAYS = [5 , 10 , 20]
    LAGS = [False , True]

    @classmethod
    def update(cls):
        for days in cls.DAYS:
            for lag1 in cls.LAGS:
                label_name = f'ret{days}' + ('_lag' if lag1 else '')
                stored_dates = PATH.db_dates(cls.DB_SRC , label_name)
                end_date     = CALENDAR.td(CALENDAR.updated() , - days - lag1)
                update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
                for date in update_dates:
                    cls.update_one(date , days , lag1 , label_name)

    @classmethod
    def update_rollback(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        for days in cls.DAYS:
            for lag1 in cls.LAGS:
                label_name = f'ret{days}' + ('_lag' if lag1 else '')
                start_date = CALENDAR.td(rollback_date , - days - lag1 + 1)
                end_date = CALENDAR.td(CALENDAR.updated() , - days - lag1)
                update_dates = CALENDAR.td_within(start_dt = start_date , end_dt = end_date)
                for date in update_dates:
                    cls.update_one(date , days , lag1 , label_name)

    @classmethod
    def update_one(cls , date : int , days : int , lag1 : bool , label_name : str):
        PATH.db_save(calc_classic_labels(date , days , lag1) , cls.DB_SRC , label_name , date , verbose = True)

def calc_classic_labels(date : int , days : int , lag1 : bool) -> Optional[pd.DataFrame]:
    d0 = CALENDAR.td(date) + lag1
    d1 = CALENDAR.td(d0 , days)

    q1 = TRADE.get_trd(d1)
    res1 = RISK.get_res(d1)
    if q1.empty or res1.empty: return

    q1 = q1.rename(columns={'adjfactor':'adj1' , 'close':'cp1'})[['secid','adj1','cp1']]
    q0 = TRADE.get_trd(d0).rename(columns={'adjfactor':'adj0' , 'close':'cp0'})[['secid','adj0','cp0']]

    ret = q1.merge(q0 , how = 'left' , on = 'secid').set_index('secid')
    label_ret = (ret['cp1'] * ret['adj1'].fillna(1) / ret['cp0'] / ret['adj0'].fillna(1) - 1).rename(f'rtn_lag{int(lag1)}_{days}')
    label_res = RISK.get_exret(d0 , int(d1)).sum().rename(f'res_lag{int(lag1)}_{days}')

    label = pd.merge(label_ret , label_res , on = 'secid').reset_index()
    return label