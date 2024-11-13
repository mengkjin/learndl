import rqdatac
import pandas as pd
import numpy as np

from typing import Literal
from datetime import datetime , timedelta

from .license_uri import uri as rcquant_uri
from ....basic import PATH
from ....func import today

START_DATE = 20241101
RC_PATH = PATH.miscel.joinpath('Rcquant')

final_path = RC_PATH.joinpath(f'min')
secdf_path = RC_PATH.joinpath('secdf')
target_path = PATH.data.joinpath('trade_ts' , 'min')

final_path.mkdir(exist_ok=True , parents=True)
secdf_path.mkdir(exist_ok=True , parents=True)

def date_offset(date : int , n = 1):
    return int((datetime.strptime(str(date), '%Y%m%d') + timedelta(days=n)).strftime('%Y%m%d'))

def rcquant_secdf(date : int):
    path = secdf_path.joinpath(f'secdf_{date}.feather')
    if path.exists(): return pd.read_feather(path)
    if not rqdatac.initialized(): rqdatac.init(uri = rcquant_uri)
    secdf = rqdatac.all_instruments(type='CS', date=str(date))
    secdf = secdf.rename(columns = {'order_book_id':'code'})
    secdf['is_active'] = secdf['status'] == 'Active'
    secdf.to_feather(path)
    return secdf

def rcquant_past_dates(file_type : Literal['secdf' , 'min']):
    path = final_path if file_type == 'min' else secdf_path
    past_files = [p for p in path.iterdir()]
    past_dates = sorted([int(p.name.split('.')[-2][-8:]) for p in past_files])
    return past_dates
    
def updated_dates():
    return PATH.db_dates('trade_ts' , 'min')
    # return rcquant_past_dates('min')

def last_date(offset : int = 0):
    dates = updated_dates()
    last_dt = max(dates) if len(dates) > 0 else date_offset(START_DATE , -1)
    return date_offset(last_dt , offset)

def trading_dates(start_date, end_date):
    if not rqdatac.initialized(): rqdatac.init(uri = rcquant_uri)
    return [int(td.strftime('%Y%m%d')) for td in rqdatac.get_trading_dates(start_date, end_date, market='cn')]

def rcquant_proceed(date : int | None = None , first_n : int = -1):
    start_date = last_date(1)
    end_date = today(-1) if date is None else date
    for dt in trading_dates(start_date , end_date):
        mark = rcquant_bar_min(dt , first_n)
        if not mark: 
            print(f'rcquant bar min {dt} failed')
            return False
        else:
            print(f'rcquant bar min {dt} success')
    return True

def rcquant_bar_min(date : int , first_n : int = -1):
    if not rqdatac.initialized(): rqdatac.init(uri = rcquant_uri)
    def code_map(x : str):
        x = x.split('.')[0]
        if x[:1] in ['3', '0']:
            y = x+'.SZ'
        elif x[:1] in ['6']:
            y = x+'.SH'
        else:
            y = x
        return y

    # date = 20240704
    sec_df = rcquant_secdf(date)
    sec_df = sec_df[sec_df['is_active']]
    if first_n > 0: sec_df = sec_df.iloc[:first_n]
    stock_list = sec_df['code'].to_numpy()
    data = rqdatac.get_price(stock_list, start_date=date, end_date=date, frequency='1m',expect_df=True)
    if isinstance(data , pd.DataFrame):
        data = data.reset_index().rename(columns = {'total_turnover':'amount', 'order_book_id':'code'}).assign(date = date)
        data['code'] = data['code'].map(code_map)
        data['time'] = data['datetime'].map(lambda x: x.strftime('%H%M%S')).str.slice(0,4)
        data['date'] = data['datetime'].map(lambda x: x.strftime('%Y%m%d'))

        data.to_feather(final_path.joinpath(f'min_bar_{date}.feather'))

        df = rcquant_min_to_normal_min(data)
        PATH.db_save(df , 'trade_ts' , 'min' , date = date , verbose = True)
        return True
    else:
        return False

def rcquant_min_to_normal_min(df : pd.DataFrame):
    df = df.copy()
    df.loc[:,['open','high','low','close','volume','amount']] = df.loc[:,['open','high','low','close','volume','amount']].astype(float)
    df['secid'] = df['code'].str.extract(r'^(\w+)\.').astype(int)
    df['minute'] = ((df['time'].str.slice(0,2).astype(int) - 9.5) * 60 + df['time'].str.slice(2,4).astype(int)).astype(int) - 1
    df.loc[df['minute'] >= 120 , 'minute'] -= 90
    df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
    df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
    df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap','num_trades']].sort_values(['secid','minute']).reset_index(drop = True)
    return df