import rqdatac
import pandas as pd
import numpy as np

from typing import Literal

from .license_uri import uri as rcquant_uri
from ....basic import PATH , CALENDAR , secid_adjust , trade_min_reform


START_DATE = 20241101
RC_PATH = PATH.miscel.joinpath('Rcquant')

final_path = RC_PATH.joinpath(f'min')
secdf_path = RC_PATH.joinpath('secdf')

final_path.mkdir(exist_ok=True , parents=True)
secdf_path.mkdir(exist_ok=True , parents=True)

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
    
def updated_dates(x_min = 1):
    assert x_min in [1 , 5 , 10 , 15 , 30 , 60]
    if x_min == 1:
        return PATH.db_dates('trade_ts' , 'min')
    else:
        return PATH.db_dates('trade_ts' , f'{x_min}min')

def min_update_dates(date):
    start_date = last_date(1)
    end_date = CALENDAR.update_to() if date is None else date
    return CALENDAR.td_within(start_date , end_date)

def x_mins_update_dates(date) -> list[int]:
    all_dates = np.array([])
    for x_min in [5 , 10 , 15 , 30 , 60]:
        source_dates = PATH.db_dates('trade_ts' , 'min')
        stored_dates = PATH.db_dates('trade_ts' , f'{x_min}min')
        dates = np.setdiff1d(CALENDAR.td_within(last_date(1 , x_min) , max(source_dates)) , stored_dates)
        all_dates = np.concatenate([all_dates , dates])
    return np.unique(all_dates).astype(int).tolist()

def x_mins_to_update(date):
    x_mins : list[int]= []
    for x_min in [5 , 10 , 15 , 30 , 60]:
        path = PATH.db_path('trade_ts' , f'{x_min}min' , date)
        if not path.exists(): x_mins.append(x_min)
    return x_mins

def last_date(offset : int = 0 , x_min : int = 1):
    dates = updated_dates(x_min)
    last_dt = max(dates) if len(dates) > 0 else CALENDAR.cd(START_DATE , -1)
    return CALENDAR.cd(last_dt , offset)

def rcquant_trading_dates(start_date, end_date):
    if not rqdatac.initialized(): rqdatac.init(uri = rcquant_uri)
    return [int(td.strftime('%Y%m%d')) for td in rqdatac.get_trading_dates(start_date, end_date, market='cn')]

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
    data = rqdatac.get_price(stock_list, start_date=str(date), end_date=str(date), frequency='1m',expect_df=True)
    if isinstance(data , pd.DataFrame) and not data.empty:
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
    df = secid_adjust(df , ['code'] , drop_old=True)
    df['minute'] = ((df['time'].str.slice(0,2).astype(int) - 9.5) * 60 + df['time'].str.slice(2,4).astype(int)).astype(int) - 1
    df.loc[df['minute'] >= 120 , 'minute'] -= 90
    df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
    df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
    df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap','num_trades']].sort_values(['secid','minute']).reset_index(drop = True)
    return df

def rcquant_proceed(date : int | None = None , first_n : int = -1):
    for dt in min_update_dates(date):
        mark = rcquant_bar_min(dt , first_n)
        if not mark: 
            print(f'rcquant bar min {dt} failed')
        else:
            print(f'rcquant bar min {dt} success')

    print('-' * 80)
    print('process other min bars:')
    for dt in x_mins_update_dates(date):
        for x_min in x_mins_to_update(dt):
            min_df = PATH.db_load('trade_ts' , 'min' , dt)
            x_min_df = trade_min_reform(min_df , x_min , 1)
            PATH.db_save(x_min_df , 'trade_ts' , f'{x_min}min' , dt , verbose = True)
    return True