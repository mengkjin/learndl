import os , platform , re
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal

from . import path_structure as PATH
from ..project import THIS_IS_SERVER

SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

DB_BY_NAME  : list[str] = ['information_js' , 'information_ts']
DB_BY_DATE  : list[str] = ['models' , 
                           'trade_js' , 'labels_js' , 'benchmark_js' , 
                           'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 
                           'sellside']  

DB_ALTERNATIVES : dict[str , str] = {
    'trade_ts' : 'trade_js' ,
    'benchmark_ts' : 'benchmark_js'
}
LOAD_PARALLEL : Literal['thread' , 'process'] | None = 'thread' if THIS_IS_SERVER else None
LOAD_MAX_WORKERS : int | Any = 40 if THIS_IS_SERVER else os.cpu_count()

deprecated_db_by_name  : list[str] = ['information_js']
deprecated_db_by_date  : list[str] = ['trade' , 'labels' , 'benchmark' , 'trade_js' , 'labels_js' , 'benchmark_js'] 

def db_src_deprecated(i):
    def wrapper(func):
        def inner(*args , **kwargs):
            db_src = args[i]
            if db_src in deprecated_db_by_name or db_src in deprecated_db_by_date:
                print(f'at {func.__name__} , {db_src} will be deprecated soon, please update your code')
            return func(*args , **kwargs)
        return inner
    return wrapper

def laptop_func_deprecated(func):
    def wrapper(*args , **kwargs):
        if THIS_IS_SERVER:
            print(f'at {func.__name__} will be deprecated soon, please update your code')
        return func(*args , **kwargs)
    return wrapper

def today(offset = 0 , astype : Any = int):
    d = datetime.today() + timedelta(days=offset)
    return astype(d.strftime('%Y%m%d'))

def list_files(directory : str | Path , fullname = False , recur = False):
    '''list all files in directory'''
    if isinstance(directory , str): directory = Path(directory)
    if recur:
        paths : list[Path] = []
        paths = [Path(dirpath).joinpath(filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames]
    else:
        paths = [p for p in directory.iterdir()]
    paths = [p.absolute() for p in paths] if fullname else [p.relative_to(directory) for p in paths]
    return paths

def dir_dates(directory : Path , start_dt = None , end_dt = None , year = None):
    target_files = list_files(directory , recur=True)
    dates = np.array(sorted(file_dates(target_files)) , dtype=int)
    if end_dt   is not None: dates = dates[dates <= (end_dt   if end_dt   > 0 else today(end_dt))]
    if start_dt is not None: dates = dates[dates >= (start_dt if start_dt > 0 else today(start_dt))]
    if year is not None:     dates = dates[dates // 10000 == year]
    return dates

def save_df(df : pd.DataFrame | None , path : Path | str , overwrite = True , printing_prefix = None):
    path = Path(path)
    if df is None or df.empty: 
        return False
    elif overwrite or not path.exists(): 
        path.parent.mkdir(parents=True , exist_ok=True)
        if SAVE_OPT_DB == 'feather':
            df.to_feather(path)
        else:
            df.to_parquet(path , engine='fastparquet')
        if printing_prefix: print(f'{printing_prefix} saved successfully')
        return True
    else:
        if printing_prefix: print(f'{printing_prefix} already exists')
        return False

def load_df(path : Path | str , raise_if_not_exist = False):
    path = Path(path)
    if not path.exists():
        if raise_if_not_exist: raise FileNotFoundError(path)
        else: return pd.DataFrame()
    if SAVE_OPT_DB == 'feather':
        df = pd.read_feather(path)
    else:
        df = pd.read_parquet(path , engine='fastparquet')
    return df

def load_df_multi(paths : dict , raise_if_not_exist = False):
    if LOAD_PARALLEL is None:
        dfs = {d:load_df(p , raise_if_not_exist=raise_if_not_exist) for d,p in paths.items()}
    else:
        assert LOAD_PARALLEL == 'thread' or platform.system() != 'Windows' , (LOAD_PARALLEL , platform.system())
        max_workers = min(LOAD_MAX_WORKERS , max(len(paths) // 5 , 1))
        PoolExecutor = ThreadPoolExecutor if LOAD_PARALLEL == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(load_df, p , raise_if_not_exist=raise_if_not_exist):d for d,p in paths.items()}
            dfs = {futures[future]:future.result() for future in as_completed(futures)}
    return dfs

def examine_df(df : pd.DataFrame , db_src , db_key , date , check_na_cols = False):
    if df.empty:
        print(f'{db_src} , {db_key} , {date} entry is empty')
    elif df.isna().all().all():
        print(f'{db_src} , {db_key} , {date} entry is all-NA')
    elif check_na_cols and df.isna().all().any():
        print(f'{db_src} , {db_key} , {date} entry has columns [{str(df.columns.values[df.isna().all()])}] all-NA')

def proc_df(raw_df : pd.DataFrame | dict[int , pd.DataFrame] , date = None, date_colname = None , check_na_cols = False , check_txt_head = 'some df' ,
            reset_index = True , ignored_fields = []):
    if isinstance(raw_df , dict):
        if date_colname: [ddf.assign(**{date_colname:d} , inplace=True) for d , ddf in raw_df.items()]
        df = pd.concat(raw_df.values())
    else:
        if date_colname and date is not None: raw_df[date_colname] = date
        df = raw_df

    if df.empty:
        print(f'{check_txt_head} is empty')
    elif df.isna().all().all():
        print(f'{check_txt_head} is all-NA')
    elif check_na_cols and df.isna().all().any():
        print(f'{check_txt_head} has columns [{str(df.columns.values[df.isna().all()])}] all-NA')

    if date_colname and date is not None: df[date_colname] = date
    if reset_index and len(df.index.names) > 1 or df.index.name: df = df.reset_index()
    if ignored_fields: df = df.drop(columns=ignored_fields , errors='ignore')
    return df

def db_path(db_src , db_key , date = None , force_type : Literal['name' , 'date'] | None = None , use_alternative = True):
    if db_src in DB_BY_NAME or force_type == 'name':
        parent = PATH.database.joinpath(f'DB_{db_src}')
        base = f'{db_key}.{SAVE_OPT_DB}'
    elif db_src in DB_BY_DATE or force_type == 'date':
        assert date is not None
        parent = PATH.database.joinpath(f'DB_{db_src}' , db_key , str(int(date) // 10000))
        base = f'{db_key}.{str(date)}.{SAVE_OPT_DB}'
    else:
        raise KeyError(db_src)
    path = parent.joinpath(base)
    if not path.exists() and use_alternative and db_src in DB_ALTERNATIVES:
        return db_path(DB_ALTERNATIVES[db_src] , db_key , date , force_type , use_alternative = False)
    else:
        return path

def db_dates(db_src , db_key , start_dt = None , end_dt = None , year = None):
    path = PATH.database.joinpath(f'DB_{db_src}' , db_key)
    return dir_dates(path , start_dt , end_dt , year)

@db_src_deprecated(1)
def db_save(df : pd.DataFrame | None , db_src , db_key , date = None , 
            force_type : Literal['name' , 'date'] | None = None , verbose = False):
    printing_prefix = f'DataBase object [{db_src}],[{db_key}],[{date}]' if verbose else None
    if df is not None and (len(df.index.names) > 1 or df.index.name): df = df.reset_index()
    mark = save_df(df , db_path(db_src , db_key , date , force_type = force_type) , overwrite = True , printing_prefix = printing_prefix)
    return mark

@db_src_deprecated(0)
def db_load(db_src , db_key , date = None , date_colname = None , 
            check_na_cols = True , raise_if_not_exist = False , ignored_fields = []) -> pd.DataFrame: 
    #  ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code']
    path = db_path(db_src , db_key , date)
    df = load_df(path , raise_if_not_exist=raise_if_not_exist)
    df = proc_df(df , date , date_colname , check_na_cols , 
                 check_txt_head=f'{db_src} , {db_key} , {date}' , 
                 ignored_fields=ignored_fields)
    return df

@db_src_deprecated(0)
def db_load_multi(db_src , db_key , dates = None , start_dt = None , end_dt = None , date_colname = None , 
                  raise_if_not_exist = False , ignored_fields = []):
    if dates is None:
        assert start_dt is not None and end_dt is not None , f'start_dt and end_dt must be provided if dates is not provided'
        dates = db_dates(db_src , db_key , start_dt , end_dt)
    paths = {date:db_path(db_src , db_key , date) for date in dates}
    dfs = load_df_multi(paths , raise_if_not_exist=raise_if_not_exist)
    df = proc_df(dfs , date_colname = date_colname , check_txt_head=f'{db_src} , {db_key} , multi dates' , ignored_fields=ignored_fields)
    return df

def pred_path(model_name : str , date : int | Any):
    return PATH.preds.joinpath(model_name , str(date // 10000) , f'{model_name}.{date}.feather')

def pred_dates(model_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.preds.joinpath(model_name) , start_dt , end_dt , year)

def pred_save(df : pd.DataFrame | None , model_name : str , date : int | Any , overwrite = True):
    return save_df(df , pred_path(model_name , date) , overwrite)

def pred_load(model_name : str , date : int | Any , date_colname = None):
    path = pred_path(model_name , date)
    df = load_df(path , raise_if_not_exist=False)
    df = proc_df(df , date , date_colname , check_txt_head=f'pred of {model_name} , {date}')
    return df

def pred_load_multi(model_name : str , dates , date_colname = None):
    paths = {date:pred_path(model_name , date) for date in dates}
    dfs = load_df_multi(paths)
    df = proc_df(dfs , date_colname = date_colname , check_txt_head=f'pred of {model_name} , multi dates')
    return df

def factor_path(factor_name : str , date : int | Any):
    return PATH.factor.joinpath(factor_name , str(date // 10000) , f'{factor_name}.{date}.feather')

def factor_dates(factor_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.factor.joinpath(factor_name) , start_dt , end_dt , year)

def factor_save(df : pd.DataFrame | None , factor_name : str , date : int | Any , overwrite = True):
    return save_df(df , factor_path(factor_name , date) , overwrite)

def factor_load(factor_name : str , date : int | Any , date_colname = None):
    path = factor_path(factor_name , date)
    df = load_df(path , raise_if_not_exist=False)
    df = proc_df(df , date , date_colname , check_txt_head=f'factor of {factor_name} , {date}')
    return df

def factor_load_multi(factor_name : str , dates , date_colname = None):
    paths = {date:factor_path(factor_name , date) for date in dates}
    dfs = load_df_multi(paths)
    df = proc_df(dfs , date_colname = date_colname , check_txt_head=f'factor of {factor_name} , multi dates')
    return df

@laptop_func_deprecated
def get_source_dates(db_src , db_key):
    assert db_src in DB_BY_DATE
    return R_source_dates(db_src , db_key)

def file_dates(path : Path | list[Path] | tuple[Path] , startswith = '' , endswith = '') -> list:
    '''get path date from R environment'''
    if isinstance(path , (list,tuple)):
        return [d[0] for d in [file_dates(p , startswith , endswith) for p in path] if d]
    else:
        if not path.name.startswith(startswith): return []
        if not path.name.endswith(endswith): return []
        s = path.stem[-8:]
        return [int(s)] if s.isdigit() else []
    
@laptop_func_deprecated
def R_dir_dates(directory):
    '''get all path dates in a dir from R environment'''
    return file_dates(list_files(directory , recur = True))
    
@laptop_func_deprecated
def R_source_dates(db_src , db_key):
    if db_src != 'benchmark_js': db_key = re.sub(r'\d+', '', db_key)
    source_key = '/'.join([db_src , db_key])
    date_source = {
        'models/risk_exp'   : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
        'models/risk_cov'   : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/6_factor_return_covariance/jm2018_model' ,
        'models/risk_spec'  : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/C_specific_risk/jm2018_model' ,
        'models/longcl_exp' : 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst',
        'trade_js/day'         : 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
        'trade_js/min'         : 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
        'labels_js/ret'        : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
        'labels_js/ret_lag'    : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
        'benchmark_js/csi300'  : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI300' ,
        'benchmark_js/csi500'  : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI500' ,
        'benchmark_js/csi1000' : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI1000' ,
    }[source_key]

    source_dates = R_dir_dates(date_source)
    return np.array(sorted(source_dates) , dtype=int)
