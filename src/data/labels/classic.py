import pandas as pd
import numpy as np

from typing import Any , Literal , Optional

from ..tushare import TSData
from ...basic import PATH

class ClassicLabelsUpdater:
    START_DATE = 20050101
    DB_SRC = 'labels_ts'

    DAYS = [5 , 10 , 20]
    LAGS = [False , True]

    @classmethod
    def proceed(cls):
        for days in cls.DAYS:
            for lag1 in cls.LAGS:
                label_name = f'ret{days}' + ('_lag' if lag1 else '')
                end_date : Any = TSData.CALENDAR.offset(TSData.CALENDAR.today , -days + 1)
                dates = TSData.CALENDAR.td_within(cls.START_DATE , end_date)
                stored_dates = PATH.db_dates(cls.DB_SRC , label_name)
                update_dates = np.setdiff1d(dates , stored_dates)
                for date in update_dates:
                    cls.update_one(date , days , lag1 , label_name)

    @classmethod
    def update_one(cls , date : int , days : int , lag1 : bool , label_name : str):
        PATH.db_save(calc_classic_labels(date , days , lag1) , cls.DB_SRC , label_name , date , verbose = True)

def calc_classic_labels(date : int , days : int , lag1 : bool) -> Optional[pd.DataFrame]:
    d0 = TSData.CALENDAR.td(date) + lag1
    d1 = TSData.CALENDAR.offset(d0 , days , 't')

    q1 = TSData.TRADE.get_trd(d1)
    res1 = TSData.MODEL.get_res(d1)
    if q1 is None or res1 is None: return

    q1 = q1.rename(columns={'adjfactor':'adj1' , 'close':'cp1'})[['secid','adj1','cp1']]
    q0 = TSData.TRADE.get_trd(d0).rename(columns={'adjfactor':'adj0' , 'close':'cp0'})[['secid','adj0','cp0']]

    ret = q1.merge(q0 , how = 'left' , on = 'secid').set_index('secid')
    label_ret = (ret['cp1'] * ret['adj1'] / ret['cp0'] / ret['adj0'] - 1).rename(f'rtn_lag{int(lag1)}_{days}')
    label_res = TSData.MODEL.get_exret(d0 , int(d1)).sum().rename(f'res_lag{int(lag1)}_{days}')

    label = pd.merge(label_ret , label_res , on = 'secid').reset_index()
    return label