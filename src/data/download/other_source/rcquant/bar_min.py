import rqdatac , re
import pandas as pd
import numpy as np

from rqdatac.share.errors import QuotaExceeded
from typing import Literal

from src.proj import PATH , MACHINE , Logger , IOCatcher
from src.basic import CALENDAR , DB
from src.data.util import secid_adjust , trade_min_reform

RC_PATH = PATH.miscel.joinpath('Rcquant')
DATA_TYPES = Literal['sec' , 'etf' , 'fut' , 'cb']
instrument_types = {'sec' : 'CS' , 'etf' : 'ETF' , 'fut' : 'Future' , 'cb' : 'Convertible'}

def src_start_date(data_type : DATA_TYPES):
    never = 20401231
    if data_type == 'sec':
        return never if MACHINE.belong_to_hfm else 20241101
    elif not MACHINE.server:
        return never
    else:
        assert data_type in ['etf' , 'fut' , 'cb'] , f'unsupported data type: {data_type}'
        return 20230601

def src_key(data_type : DATA_TYPES , x_min : int = 1):
    if data_type == 'sec':
        prefix = ''
    else:
        prefix = f'{data_type}_'
    if x_min == 1:
        return f'{prefix}min'
    else:
        return f'{prefix}{x_min}min'

def load_list(date : int , data_type : DATA_TYPES):
    path = RC_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    if path.exists(): 
        return pd.read_feather(path)
    return None

def write_list(df : pd.DataFrame , date : int , data_type : DATA_TYPES):
    path = RC_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    df.to_feather(path)

def load_min(date : int , data_type : DATA_TYPES):
    path = RC_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    if path.exists(): 
        return pd.read_feather(path)
    return None

def write_min(df : pd.DataFrame , date : int , data_type : DATA_TYPES):
    path = RC_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    df.to_feather(path)

def rcquant_init():
    if not rqdatac.initialized(): 
        try:
            with IOCatcher() as catcher:
                rcquant_uri = MACHINE.local_settings('rcquant')['uri']
                rqdatac.init(uri = rcquant_uri)
            output = catcher.contents
            if _print := output['stdout']:
                print(_print)
            if _error := output['stderr']:
                key_info = re.search(r'Your account will be expired after  (\d+) days', _error)
                if key_info:
                    Logger.warning(f'RcQuant Warning : {key_info.group(0)}')
                else:
                    Logger.error(f'RcQuant Error : {_error}')
        except FileNotFoundError:
            Logger.error(f'rcquant login info not found, please check .local_settings/rcquant.yaml')
            return False
        except QuotaExceeded as e:
            Logger.error(f'rcquant init failed: {e}')
            return False
    return True

def rcquant_past_dates(data_type : DATA_TYPES , file_type : Literal['secdf' , 'min']):
    path = RC_PATH.joinpath(f'{data_type}min') if file_type == 'min' else RC_PATH.joinpath(f'{data_type}df')
    past_files = [p for p in path.iterdir()]
    past_dates = sorted([int(p.name.split('.')[-2][-8:]) for p in past_files])
    return past_dates
    
def stored_dates(data_type : DATA_TYPES , x_min : int = 1):
    assert x_min in [1 , 5 , 10 , 15 , 30 , 60] , f'only support 1min , 5min , 10min , 15min , 30min , 60min : {x_min}'
    if x_min != 1:
        assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
    return DB.dates('trade_ts' , src_key(data_type , x_min) , use_alt = False)

def last_date(data_type : DATA_TYPES , offset : int = 0 , x_min : int = 1):
    dates = stored_dates(data_type , x_min)
    last_dt = max(dates) if len(dates) > 0 else 19970101
    return CALENDAR.cd(last_dt , offset)

def target_dates(data_type : DATA_TYPES , date : int | None = None):
    start = src_start_date(data_type)
    end   = CALENDAR.update_to() if date is None else date
    dates = CALENDAR.td_within(start , end)
    return CALENDAR.diffs(dates , stored_dates(data_type , 1))

def x_mins_target_dates(data_type : DATA_TYPES , date : int | None = None) -> list[int]:
    if data_type != 'sec': 
        return []
    all_dates = np.array([])
    for x_min in [5 , 10 , 15 , 30 , 60]:
        source_dates = DB.dates('trade_ts' , src_key(data_type , 1))
        stored_dates = DB.dates('trade_ts' , src_key(data_type , x_min))
        target_dates = CALENDAR.diffs(source_dates , stored_dates)
        dates = target_dates[target_dates >= src_start_date(data_type)]
        all_dates = np.concatenate([all_dates , dates])
    end = CALENDAR.update_to() if date is None else date
    all_dates = np.unique(all_dates[all_dates <= end]).astype(int).tolist()
    return all_dates

def x_mins_to_update(date , data_type : DATA_TYPES):
    if data_type != 'sec': 
        return []
    x_mins : list[int]= []
    for x_min in [5 , 10 , 15 , 30 , 60]:
        path = DB.path('trade_ts' , src_key(data_type , x_min) , date)
        if not path.exists(): 
            x_mins.append(x_min)
    return x_mins

def rcquant_instrument_list(date : int , data_type : DATA_TYPES):
    secdf = load_list(date , data_type)
    if secdf is not None: 
        return secdf
    if not rcquant_init(): 
        return pd.DataFrame()
    secdf = rqdatac.all_instruments(type=instrument_types[data_type], date=str(date))
    secdf = secdf.rename(columns = {'order_book_id':'code'})
    if 'status' in secdf.columns:
        secdf['is_active'] = secdf['status'] == 'Active'
    else:
        secdf['is_active'] = True
    write_list(secdf , date , data_type)
    return secdf

def rcquant_trading_dates(start_date, end_date):
    if not rcquant_init(): 
        return []
    return [int(td.strftime('%Y%m%d')) for td in rqdatac.get_trading_dates(start_date, end_date, market='cn')]

def rcquant_bar_min(date : int , data_type : DATA_TYPES , first_n : int = -1):    
    def code_map(x : str):
        if data_type != 'sec': 
            return x
        x = x.split('.')[0]
        if x[:1] in ['3', '0']:
            y = x+'.SZ'
        elif x[:1] in ['6']:
            y = x+'.SH'
        else:
            y = x
        return y

    if (sec_min := load_min(date , data_type)) is not None: 
        df = rcquant_min_to_normal_min(sec_min , data_type)
        DB.save(df , 'trade_ts' , src_key(data_type) , date = date , verbose = True)
        return True

    if not rcquant_init(): 
        return False

    instrument_list = rcquant_instrument_list(date , data_type = data_type)
    instrument_list = instrument_list.loc[instrument_list['is_active']]
    if first_n > 0: 
        instrument_list = instrument_list.iloc[:first_n]
    code_list = instrument_list['code'].to_numpy(str)
    data = rqdatac.get_price(code_list, start_date=str(date), end_date=str(date), frequency='1m',expect_df=True)
    if isinstance(data , pd.DataFrame) and not data.empty:
        data = data.reset_index().rename(columns = {'total_turnover':'amount', 'order_book_id':'code'}).assign(date = date)
        data['code'] = data['code'].map(code_map)
        data['time'] = data['datetime'].map(lambda x: x.strftime('%H%M%S')).str.slice(0,4)
        data['date'] = data['datetime'].map(lambda x: x.strftime('%Y%m%d'))

        write_min(data , date , data_type)

        df = rcquant_min_to_normal_min(data , data_type)
        DB.save(df , 'trade_ts' , src_key(data_type) , date = date , verbose = True)
        return True
    else:
        return False

def rcquant_min_to_normal_min(df : pd.DataFrame , data_type : DATA_TYPES):
    if data_type != 'sec': 
        return df
    df = df.copy()
    df.loc[:,['open','high','low','close','volume','amount']] = df.loc[:,['open','high','low','close','volume','amount']].astype(float)
    df = secid_adjust(df , ['code'] , drop_old=True)
    df['minute'] = ((df['time'].str.slice(0,2).astype(int) - 9.5) * 60 + df['time'].str.slice(2,4).astype(int)).astype(int) - 1
    df.loc[df['minute'] >= 120 , 'minute'] -= 90
    df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
    df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
    df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap','num_trades']].sort_values(['secid','minute']).reset_index(drop = True)
    return df

def rcquant_download(date : int | None = None , data_type : DATA_TYPES | None = None ,  first_n : int = -1):
    assert data_type is not None , f'data_type is required'
    dts = target_dates(data_type , date)
    if len(dts) == 0: 
        print(f'Skipping: rcquant {data_type} bar min has no date to download')
    for dt in dts:
        mark = rcquant_bar_min(dt , data_type , first_n)
        if not mark: 
            Logger.fail(f'rcquant {data_type} bar min {dt} failed')
        else:
            Logger.success(f'rcquant {data_type} bar min {dt} success')

    for dt in x_mins_target_dates(data_type , date):
        print(f'process other {data_type} min bars at {dt} from source rcquant')
        for x_min in x_mins_to_update(dt , data_type = data_type):
            min_df = DB.load('trade_ts' , src_key(data_type) , dt)
            assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
            x_min_df = trade_min_reform(min_df , x_min , 1)
            DB.save(x_min_df , 'trade_ts' , src_key(data_type , x_min) , dt , verbose = True)
        Logger.divider()
    return True

def rcquant_proceed(date : int | None = None , first_n : int = -1):

    data_types : list[DATA_TYPES] = ['sec' , 'etf' , 'fut' , 'cb']
    for data_type in data_types:
        try:
            rcquant_download(date , data_type , first_n)
        except Exception as e:
            Logger.error(f'rcquant download {data_type} minbar failed: {e}')
            return False
    
    return True