import baostock as bs
import numpy as np
import pandas as pd  

from typing import Any , Literal

from src.proj import PATH , Logger
from src.basic import CALENDAR , DB
from src.data.util import secid_adjust , trade_min_reform

START_DATE = 20401231
BAO_PATH = PATH.miscel.joinpath('Baostock')

final_path = BAO_PATH.joinpath(f'5min')
secdf_path = BAO_PATH.joinpath('secdf')
task_path = BAO_PATH.joinpath('task')

final_path.mkdir(exist_ok=True , parents=True)
secdf_path.mkdir(exist_ok=True , parents=True)
task_path.mkdir(exist_ok=True , parents=True)

def tmp_file_dir(start_dt : int , end_dt : int):
    path = task_path.joinpath(f'{start_dt}_{end_dt}')
    return path

def tmp_file_path(start_dt : int , end_dt : int , code : str):
    path = task_path.joinpath(f'{start_dt}_{end_dt}' , str(code))
    return path

def baostock_secdf(date : int):
    path = secdf_path.joinpath(f'secdf_{date}.feather')
    if path.exists():
        return DB.load_df(path)
    end_date_str = f'{date // 10000}-{(date // 100) % 100}-{date % 100}'
    secdf = bs.query_all_stock(end_date_str).get_data()
    secdf['market'] = secdf['code'].str.slice(0,2)
    secdf['secid'] = secdf['code'].str.slice(3).astype(int)
    secdf['is_sec'] = ((secdf['market'] == 'sh') * (secdf['secid'] >= 600000) + \
        (secdf['market'] == 'sz') * (secdf['secid'] < 600000)) * (~secdf['code_name'].str.contains('指数'))
    DB.save_df(secdf , path , verbose = False)
    return secdf

def baostock_past_dates(file_type : Literal['secdf' , '5min']):
    path = final_path if file_type == '5min' else secdf_path
    past_files = [p for p in path.iterdir()]
    past_dates = sorted([int(p.name.split('.')[-2][-8:]) for p in past_files])
    return past_dates
    
def updated_dates(x_min : int = 5):
    assert x_min in [5 , 10 , 15 , 30 , 60] , f'{x_min} is not in [5 , 10 , 15 , 30 , 60]'
    return DB.dates('trade_ts' , f'{x_min}min')

def updatable(date , last_date):
    return (len(updated_dates()) == 0) or (date > 0 and date > CALENDAR.cd(last_date , 6))

def x_mins_update_dates(date) -> list[int]:
    all_dates = np.array([])
    for x_min in [10 , 15 , 30 , 60]:
        source_dates = DB.dates('trade_ts' , '5min')
        stored_dates = DB.dates('trade_ts' , f'{x_min}min')
        dates = CALENDAR.diffs(last_date_x_min(1 , x_min) , max(source_dates) , stored_dates)
        all_dates = np.concatenate([all_dates , dates])
    return np.unique(all_dates).astype(int).tolist()

def x_mins_to_update(date):
    x_mins : list[int] = []
    for x_min in [10 , 15 , 30 , 60]:
        path = DB.path('trade_ts' , f'{x_min}min' , date)
        if not path.exists(): 
            x_mins.append(x_min)
    return x_mins

def last_date_x_min(offset : int = 0 , x_min : int = 10):
    dates = updated_dates(x_min)
    last_dt = max(dates) if len(dates) > 0 else CALENDAR.cd(START_DATE , -1)
    return CALENDAR.cd(last_dt , offset)

def last_date(offset : int = 0):
    dates = baostock_past_dates('5min')
    last_dt = max(dates) if len(dates) > 0 else CALENDAR.cd(START_DATE , -1)

    target_last_dt = last_date_x_min(0 , 5)
    if target_last_dt > last_dt: 
        last_dt = target_last_dt
    return CALENDAR.cd(last_dt , offset)

def pending_date():
    dates0 = baostock_past_dates('secdf')
    d0 = max(dates0) if len(dates0) > 0 else CALENDAR.cd(START_DATE , -1)
    d1 = last_date()
    return d0 if d0 > d1 else -1

def baostock_bar_5min(start_dt : int , end_dt : int , first_n : int = -1 , retry_n : int = 10):
    # date = 20240704
    if end_dt < start_dt: 
        return True
    retry = 0
    
    tmp_dir = tmp_file_dir(start_dt , end_dt)
    tmp_dir.mkdir(exist_ok=True)

    start_date_str = f'{start_dt // 10000}-{(start_dt // 100) % 100}-{start_dt % 100}'
    end_date_str = f'{end_dt // 10000}-{(end_dt // 100) % 100}-{end_dt % 100}'
    
    sec_df : pd.DataFrame | Any = None
    while True:
        if retry >= retry_n: 
            return False
        try:
            bs.login()
            if sec_df is None:
                sec_df = baostock_secdf(end_dt).query('is_sec == 1')
                if first_n > 0: 
                    sec_df = sec_df.iloc[:first_n]
            downloaded = [d.name for d in tmp_dir.iterdir()]
            task_codes = np.setdiff1d(sec_df['code'].to_numpy() , downloaded)
            if len(task_codes) == 0: 
                bs.logout()
                break
            
            Logger.stdout(f'{start_dt} - {end_dt} : {len(downloaded)} already downloaded , {len(task_codes)} codes to download :')
            for i , code in enumerate(task_codes):
                rs = bs.query_history_k_data_plus(code, 'date,time,code,open,high,low,close,volume,amount,adjustflag',
                                                  start_date=start_date_str,end_date=end_date_str,frequency='5', adjustflag='3')
                assert rs is not None , f'{rs} is None , corrupted data'
                result = rs.get_data()
                if isinstance(result , pd.DataFrame):
                    DB.save_df(result , tmp_file_path(start_dt , end_dt , code) , verbose = False)

                if i % 100 == 0:
                    Logger.success(f'{i + 1}/{len(task_codes)} {start_dt} - {end_dt} : {code}...' , end = '\r')

        except Exception as e:
            bs.logout()
            Logger.error(f'Baostock 5min download failed at {start_dt} - {end_dt} : {code} , retry {retry} : {e}')
            retry += 1
        else:
            break

    df_list = [DB.load_df(d) for d in tmp_dir.iterdir()]
    if len(df_list) == 0: 
        return False
    df_all = pd.concat([DB.load_df(d) for d in tmp_dir.iterdir()])

    for date_str in df_all['date'].unique():
        df : pd.DataFrame = df_all.query('date == @date_str')
        date = int(str(date_str).replace('-', ''))
        df = df.copy().reset_index(drop = True).assign(date = date)
        DB.save_df(df , final_path.joinpath(f'5min_bar_{date}.feather') , verbose = False)

        df = baostock_5min_to_normal_5min(df)
        DB.save(df , 'trade_ts' , '5min' , date = date , verbose = True)
    # del after : No!
    '''
    if first_n <= 0:
        [d.unlink() for d in tmp_dir.iterdir()]
        tmp_dir.unlink()
    '''
        
    return True

def baostock_5min_to_normal_5min(df : pd.DataFrame):
    df.loc[:,['open','high','low','close','volume','amount']] = df.loc[:,['open','high','low','close','volume','amount']].astype(float)
    df = secid_adjust(df , ['code'] , drop_old=True)
    df['minute'] = ((df['time'].str.slice(8,10).astype(int) - 9.5) * 12 + df['time'].str.slice(10,12).astype(int) / 5).astype(int) - 1
    df.loc[df['minute'] >= 24 , 'minute'] -= 18
    df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
    df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
    df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap']].sort_values(['secid','minute']).reset_index(drop = True)
    return df

def baostock_proceed(date : int | None = None , first_n : int = -1 , retry_n : int = 10 , verbosity : int = 1):
    pending_dt = pending_date()
    end_dt = CALENDAR.update_to() if date is None else date
    if pending_dt == end_dt: 
        pending_dt = -1
    for dt in [pending_dt , end_dt]:
        last_dt = last_date(1)
        if (updatable(dt , last_dt) or (date == dt)) and (dt >= last_dt):
            mark = baostock_bar_5min(last_dt , dt , first_n , retry_n)
            if not mark: 
                Logger.alert(f'baostock 5min {last_dt} - {dt} failed')
            elif verbosity > 1 :
                Logger.success(f'baostock 5min {last_dt} - {dt} success')

    for dt in x_mins_update_dates(date):
        Logger.stdout(f'Transform: sec X-min bars at {dt} from source baostock')
        for x_min in x_mins_to_update(dt):
            five_min_df = DB.load('trade_ts' , '5min' , dt)
            x_min_df = trade_min_reform(five_min_df , x_min , 5)
            DB.save(x_min_df , 'trade_ts' , f'{x_min}min' , dt , verbose = True)

    return True