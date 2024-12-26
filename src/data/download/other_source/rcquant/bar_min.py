import rqdatac
import pandas as pd
import numpy as np

from rqdatac.share.errors import QuotaExceeded
from typing import Literal

from src.basic import PATH , CALENDAR , CONF , IS_SERVER , MY_SERVER , MY_LAPTOP
from src.data.util.basic import secid_adjust , trade_min_reform
from src.func.display import print_seperator

SEC_START_DATE = 20241101
ETF_START_DATE = 20200101 if IS_SERVER else 20241224
FUT_START_DATE = 20200101 if IS_SERVER else 20241224

RC_PATH = PATH.miscel.joinpath('Rcquant')

def src_start_date(data_type : Literal['sec' , 'etf' , 'fut']):
    if MY_SERVER or MY_LAPTOP:
        if data_type == 'sec':
            return SEC_START_DATE
        elif data_type == 'etf':
            return ETF_START_DATE
        elif data_type == 'fut':
            return FUT_START_DATE
    else:
        return 20401231

def src_key(data_type : Literal['sec' , 'etf' , 'fut'] , x_min : int = 1):
    if data_type == 'sec':
        prefix = ''
    else:
        prefix = f'{data_type}_'
    if x_min == 1:
        return f'{prefix}min'
    else:
        return f'{prefix}{x_min}min'

def load_list(date : int , data_type : Literal['sec' , 'etf' , 'fut']):
    path = RC_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    if path.exists(): return pd.read_feather(path)
    return None

def write_list(df : pd.DataFrame , date : int , data_type : Literal['sec' , 'etf' , 'fut']):
    path = RC_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    df.to_feather(path)

def load_min(date : int , data_type : Literal['sec' , 'etf' , 'fut']):
    path = RC_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    if path.exists(): return pd.read_feather(path)
    return None

def write_min(df : pd.DataFrame , date : int , data_type : Literal['sec' , 'etf' , 'fut']):
    path = RC_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    df.to_feather(path)

def rcquant_license_check():
    if 'rcquant.yaml' not in [p.name for p in PATH.conf.joinpath('confidential').rglob('*.yaml')]:
        print(f'rcquant login info not found, please check configs/confidential/rcquant.yaml')
        return False
    return True

def rcquant_init():
    if not rqdatac.initialized(): 
        rcquant_uri = CONF.confidential('rcquant')['uri']
        try:
            rqdatac.init(uri = rcquant_uri)
        except QuotaExceeded as e:
            print(f'rcquant init failed: {e}')
            return False
    return True

def rcquant_past_dates(data_type : Literal['sec' , 'etf' , 'fut'] , file_type : Literal['secdf' , 'min']):
    path = RC_PATH.joinpath(f'{data_type}min') if file_type == 'min' else RC_PATH.joinpath(f'{data_type}df')
    past_files = [p for p in path.iterdir()]
    past_dates = sorted([int(p.name.split('.')[-2][-8:]) for p in past_files])
    return past_dates
    
def updated_dates(x_min = 1 , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    assert data_type in ['sec' , 'etf' , 'fut'] , f'only support sec , etf , fut : {data_type}'
    assert x_min in [1 , 5 , 10 , 15 , 30 , 60] , f'only support 1min , 5min , 10min , 15min , 30min , 60min : {x_min}'
    if x_min != 1:
        assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
    return PATH.db_dates('trade_ts' , src_key(data_type , x_min))

def last_date(offset : int = 0 , x_min : int = 1 , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    dates = updated_dates(x_min , data_type = data_type)
    last_dt = max(dates) if len(dates) > 0 else 19970101
    return CALENDAR.cd(last_dt , offset)

def min_update_dates(date , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    assert data_type in ['sec' , 'etf' , 'fut'] , f'only support sec , etf , fut : {data_type}'
    start = last_date(1 , data_type = data_type)
    end   = CALENDAR.update_to() if date is None else date
    dates = CALENDAR.td_within(start , end)
    return dates[dates >= src_start_date(data_type)]

def x_mins_update_dates(date , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec') -> list[int]:
    if data_type != 'sec': return []
    all_dates = np.array([])
    for x_min in [5 , 10 , 15 , 30 , 60]:
        source_dates = PATH.db_dates('trade_ts' , src_key(data_type , 1))
        stored_dates = PATH.db_dates('trade_ts' , src_key(data_type , x_min))
        dates = CALENDAR.diffs(last_date(1 , x_min) , max(source_dates) , stored_dates)
        all_dates = np.concatenate([all_dates , dates])
    return np.unique(all_dates).astype(int).tolist()

def x_mins_to_update(date , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    if data_type != 'sec': return []
    x_mins : list[int]= []
    for x_min in [5 , 10 , 15 , 30 , 60]:
        path = PATH.db_path('trade_ts' , src_key(data_type , x_min) , date)
        if not path.exists(): x_mins.append(x_min)
    return x_mins

def rcquant_instrument_list(date : int , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    secdf = load_list(date , data_type)
    if secdf is not None: return secdf
    rcquant_init()
    instrument_types = {'sec' : 'CS' , 'etf' : 'ETF' , 'fut' : 'Future'}
    secdf = rqdatac.all_instruments(type=instrument_types[data_type], date=str(date))
    secdf = secdf.rename(columns = {'order_book_id':'code'})
    if 'status' in secdf.columns:
        secdf['is_active'] = secdf['status'] == 'Active'
    else:
        secdf['is_active'] = True
    write_list(secdf , date , data_type)
    return secdf

def rcquant_trading_dates(start_date, end_date):
    rcquant_init()
    return [int(td.strftime('%Y%m%d')) for td in rqdatac.get_trading_dates(start_date, end_date, market='cn')]

def rcquant_bar_min(date : int , first_n : int = -1 , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    if not rcquant_license_check(): return False
    
    def code_map(x : str):
        if data_type != 'sec': return x
        x = x.split('.')[0]
        if x[:1] in ['3', '0']:
            y = x+'.SZ'
        elif x[:1] in ['6']:
            y = x+'.SH'
        else:
            y = x
        return y

    if (sec_min := load_min(date , data_type)) is not None: 
        df = rcquant_min_to_normal_min(sec_min)
        PATH.db_save(df , 'trade_ts' , src_key(data_type) , date = date , verbose = True)
        return True

    rcquant_init()

    instrument_list = rcquant_instrument_list(date , data_type = data_type)
    instrument_list = instrument_list[instrument_list['is_active']]
    if first_n > 0: instrument_list = instrument_list.iloc[:first_n]
    code_list = instrument_list['code'].to_numpy()
    data = rqdatac.get_price(code_list, start_date=str(date), end_date=str(date), frequency='1m',expect_df=True)
    if isinstance(data , pd.DataFrame) and not data.empty:
        data = data.reset_index().rename(columns = {'total_turnover':'amount', 'order_book_id':'code'}).assign(date = date)
        data['code'] = data['code'].map(code_map)
        data['time'] = data['datetime'].map(lambda x: x.strftime('%H%M%S')).str.slice(0,4)
        data['date'] = data['datetime'].map(lambda x: x.strftime('%Y%m%d'))

        write_min(data , date , data_type)

        df = rcquant_min_to_normal_min(data)
        PATH.db_save(df , 'trade_ts' , src_key(data_type) , date = date , verbose = True)
        return True
    else:
        return False

def rcquant_min_to_normal_min(df : pd.DataFrame , data_type : Literal['sec' , 'etf' , 'fut'] = 'sec'):
    if data_type != 'sec': return df
    df = df.copy()
    df.loc[:,['open','high','low','close','volume','amount']] = df.loc[:,['open','high','low','close','volume','amount']].astype(float)
    df = secid_adjust(df , ['code'] , drop_old=True)
    df['minute'] = ((df['time'].str.slice(0,2).astype(int) - 9.5) * 60 + df['time'].str.slice(2,4).astype(int)).astype(int) - 1
    df.loc[df['minute'] >= 120 , 'minute'] -= 90
    df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
    df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
    df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap','num_trades']].sort_values(['secid','minute']).reset_index(drop = True)
    return df

def rcquant_download(date : int | None = None ,data_type : Literal['sec' , 'etf' , 'fut'] = 'sec' ,  first_n : int = -1):
    for dt in min_update_dates(date , data_type = data_type):
        mark = rcquant_bar_min(dt , first_n , data_type = data_type)
        if not mark: 
            print(f'rcquant {data_type} bar min {dt} failed')
        else:
            print(f'rcquant {data_type} bar min {dt} success')

    for dt in x_mins_update_dates(date , data_type = data_type):
        print(f'process other {data_type} min bars at {dt} from source rcquant')
        for x_min in x_mins_to_update(dt , data_type = data_type):
            min_df = PATH.db_load('trade_ts' , src_key(data_type) , dt)
            assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
            x_min_df = trade_min_reform(min_df , x_min , 1)
            PATH.db_save(x_min_df , 'trade_ts' , src_key(data_type , x_min) , dt , verbose = True)
        print_seperator()
    return True

def rcquant_proceed(date : int | None = None , first_n : int = -1):
    if not rcquant_license_check(): return False
    if not rcquant_init(): return False

    try:
        rcquant_download(date , 'sec' , first_n)
    except Exception as e:
        print(f'rcquant download sec minbar failed: {e}')
        return False
    try:
        rcquant_download(date , 'etf' , first_n)
    except Exception as e:
        print(f'rcquant download etf minbar failed: {e}')
        return False
    try:
        rcquant_download(date , 'fut' , first_n)
    except Exception as e:
        print(f'rcquant download fut minbar failed: {e}')
        return False
    return True