import pandas as pd
import numpy as np

from typing import Any

from src.basic import PATH , CALENDAR

class MultiKlineUpdater:
    START_DATE = 20050101
    DB_SRC = 'trade_ts'

    DAYS = [5 , 10 , 20]

    @classmethod
    def update(cls):
        for n_day in cls.DAYS:
            label_name = f'{n_day}day'
            stored_dates = PATH.db_dates(cls.DB_SRC , label_name)
            end_date     = PATH.db_dates('trade_ts' , 'day').max()
            update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
            for date in update_dates: cls.update_one(date , n_day , label_name)

    @classmethod
    def update_one(cls , date : int , n_day : int , label_name : str):
        PATH.db_save(nday_kline(date , n_day) , cls.DB_SRC , label_name , date , verbose = True)


def nday_kline(date : int , n_day : int):
    '''from day to n_day'''
    # read calendar
    assert n_day in [5 , 10 , 20] , f'n_day should be in [5 , 10 , 20]'
    trailing_dates = CALENDAR.td_trailing(date , n_day)
    assert trailing_dates[-1] == date , (trailing_dates[-1] , date)

    price_feat  = ['open','close','high','low','vwap']
    volume_feat = ['amount','volume','turn_tt','turn_fl','turn_fr']

    datas = [PATH.db_load('trade_ts' , 'day' , d , date_colname='date') for d in trailing_dates]
    datas = [d for d in datas if not d.empty]
    if not datas: return pd.DataFrame()
    with np.errstate(invalid='ignore' , divide = 'ignore'):
        data = pd.concat(datas , axis = 0).sort_values(['secid','date'])
        data.loc[:,'adjfactor'] = data.loc[:,'adjfactor'].ffill().fillna(1)
        data.loc[:,price_feat] = data.loc[:,price_feat] * data.loc[:,'adjfactor'].values[:,None]
        data['pctchange'] = data['pctchange'] / 100 + 1
        data['vwap'] = data['vwap'] * data['volume']
        agg_dict = {'open':'first','high':'max','low':'min','close':'last','pctchange':'prod','vwap':'sum',**{k:'sum' for k in volume_feat},}
        df = data.groupby('secid').agg(agg_dict)
        df['pctchange'] = (df['pctchange'] - 1) * 100
        df['vwap'] /= np.where(df['volume'] == 0 , np.nan , df['volume'])
        df['vwap'] = df['vwap'].where(~df['vwap'].isna() , df['close'])
    return df
