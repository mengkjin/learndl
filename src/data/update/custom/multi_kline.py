import pandas as pd
import numpy as np
from typing import Any , Literal
from src.proj import CALENDAR , DB , Logger

from src.data.update.custom.basic import BasicCustomUpdater

class MultiKlineUpdater(BasicCustomUpdater):
    START_DATE = 20050101
    DB_SRC = 'trade_ts'

    DAYS = [5 , 10 , 20]

    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : int = 1):
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all nday klines is not supported yet for {cls.__name__}')
        
        for n_day in cls.DAYS:
            label_name = f'{n_day}day'
            if update_type == 'recalc':
                stored_dates = np.array([])
            elif update_type == 'update':
                stored_dates = DB.dates(cls.DB_SRC , label_name)
            elif update_type == 'rollback':
                rollback_date = CALENDAR.td(cls._rollback_date , 1)
                stored_dates = CALENDAR.slice(DB.dates(cls.DB_SRC , label_name) , 0 , rollback_date - 1)
            else:
                raise ValueError(f'Invalid update type: {update_type}')
            end_date     = DB.dates(cls.DB_SRC , 'day').max()
            update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
            if len(update_dates) == 0:
                Logger.skipping(f'{cls.DB_SRC}/{label_name} is up to date' , indent = indent , vb_level = vb_level)
                continue

            for date in update_dates: 
                cls.update_one(date , n_day , label_name , indent = indent + 1 , vb_level = vb_level + 2)
            Logger.success(f'Update {cls.DB_SRC}/{label_name} at {CALENDAR.dates_str(update_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , n_day : int , label_name : str , indent : int = 2 , vb_level : int = 2):
        DB.save(nday_kline(date , n_day) , cls.DB_SRC , label_name , date , indent = indent , vb_level = vb_level)


def nday_kline(date : int , n_day : int) -> pd.DataFrame:
    '''from day to n_day'''
    # read calendar
    assert n_day in [5 , 10 , 20] , f'n_day should be in [5 , 10 , 20]'
    trailing_dates = CALENDAR.td_trailing(date , n_day)
    assert trailing_dates[-1] == date , (trailing_dates[-1] , date)

    price_feat  = ['open','close','high','low','vwap']
    volume_feat = ['amount','volume','turn_tt','turn_fl','turn_fr']

    datas = [DB.load('trade_ts' , 'day' , d , date_colname='date') for d in trailing_dates]
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
