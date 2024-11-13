import os , platform , re
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal

from . import project as PATH

SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

DB_BY_NAME  : list[str] = ['information' , 'information_ts']
DB_BY_DATE  : list[str] = ['models' , 
                           'trade' , 'labels' , 'benchmark' , 'membership_ts' , 
                           'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' ,
                           'sellside']  

deprecated_db_by_name  : list[str] = ['information']
deprecated_db_by_date  : list[str] = ['trade' , 'labels' , 'benchmark'] 

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
        if PATH.THIS_IS_SERVER:
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

def save_df(df : pd.DataFrame | None , target_path : Path | str , overwrite = True , printing_prefix = None):
    target_path = Path(target_path)
    if df is None or df.empty or len(df) == 0: 
        return False
    elif overwrite or not target_path.exists(): 
        target_path.parent.mkdir(parents=True , exist_ok=True)
        if SAVE_OPT_DB == 'feather':
            df.to_feather(target_path)
        else:
            df.to_parquet(target_path , engine='fastparquet')
        if printing_prefix: print(f'{printing_prefix} saved successfully')
        return True
    else:
        if printing_prefix: print(f'{printing_prefix} already exists')
        return False

def load_df(target_path : Path | str , none_if_not_exist = False):
    target_path = Path(target_path)
    if not target_path.exists():
        if none_if_not_exist: return None
        raise FileNotFoundError(target_path)
    if SAVE_OPT_DB == 'feather':
        return pd.read_feather(target_path)
    else:
        return pd.read_parquet(target_path , engine='fastparquet')

@db_src_deprecated(1)
def db_save(df : pd.DataFrame | None , db_src , db_key , date = None , 
            force_type : Literal['name' , 'date'] | None = None , verbose = False):
    printing_prefix = f'DataBase object [{db_src}],[{db_key}],[{date}]' if verbose else None
    mark = save_df(df , db_path(db_src , db_key , date , force_type = force_type) , overwrite = True , printing_prefix = printing_prefix)
    return mark

@db_src_deprecated(0)
def db_load(db_src , db_key , date = None , check_na_cols = True) -> pd.DataFrame | Any:
    path = db_path(db_src , db_key , date)
    df = load_df(path , none_if_not_exist=True)
    if df is None: return None
    if df.empty:
        print(f'{db_src} , {db_key} , {date} entry is empty')
    elif df.isna().all().all():
        print(f'{db_src} , {db_key} , {date} entry is all-NA')
    elif check_na_cols and df.isna().all().any():
        print(f'{db_src} , {db_key} , {date} entry has columns [{str(df.columns.values[df.isna().all()])}] all-NA')
    return df

@db_src_deprecated(0)
def db_load_multi(db_src , db_key , dates , parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
    if parallel is None:
        dfs = {date:db_load(db_src , db_key , date) for date in dates}
    else:
        assert parallel == 'thread' or platform.system() != 'Windows' , (parallel , platform.system())
        if n_cpu:= os.cpu_count(): max_workers = min(max_workers , n_cpu)
        PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(db_load, db_src , db_key , date):date for date in dates}
            dfs = {futures[future]:future.result() for future in as_completed(futures)}
    return dfs

def db_path(db_src , db_key , date = None , force_type : Literal['name' , 'date'] | None = None):
    if db_src in DB_BY_NAME or force_type == 'name':
        db_path = PATH.database.joinpath(f'DB_{db_src}')
        db_base = f'{db_key}.{SAVE_OPT_DB}'
    elif db_src in DB_BY_DATE or force_type == 'date':
        assert date is not None
        db_path = PATH.database.joinpath(f'DB_{db_src}' , db_key , str(int(date) // 10000))
        db_base = f'{db_key}.{str(date)}.{SAVE_OPT_DB}'
    else:
        raise KeyError(db_src)
    return db_path.joinpath(db_base)

def db_dates(db_src , db_key , start_dt = None , end_dt = None , year = None):
    db_path = PATH.database.joinpath(f'DB_{db_src}' , db_key)
    return dir_dates(db_path , start_dt , end_dt , year)

def pred_path(model_name : str , date : int | Any):
    return PATH.preds.joinpath(model_name , str(date // 10000) , f'{model_name}.{date}.feather')

def pred_dates(model_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.preds.joinpath(model_name) , start_dt , end_dt , year)

def pred_save(df : pd.DataFrame | None , model_name : str , date : int | Any , overwrite = True):
    return save_df(df , pred_path(model_name , date) , overwrite)

def pred_load(model_name : str , date : int | Any):
    return load_df(pred_path(model_name , date) , none_if_not_exist=True)

def factor_path(factor_name : str , date : int | Any):
    return PATH.factor.joinpath(factor_name , str(date // 10000) , f'{factor_name}.{date}.feather')

def factor_dates(factor_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.factor.joinpath(factor_name) , start_dt , end_dt , year)

def factor_save(df : pd.DataFrame | None , factor_name : str , date : int | Any , overwrite = True):
    return save_df(df , factor_path(factor_name , date) , overwrite)

def factor_load(factor_name : str , date : int | Any):
    return load_df(factor_path(factor_name , date) , none_if_not_exist=True)

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
    if db_src != 'benchmark': db_key = re.sub(r'\d+', '', db_key)
    source_key = '/'.join([db_src , db_key])
    date_source = {
        'models/risk_exp'   : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
        'models/risk_cov'   : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/6_factor_return_covariance/jm2018_model' ,
        'models/risk_spec'  : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/C_specific_risk/jm2018_model' ,
        'models/longcl_exp' : 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst',
        'trade/day'         : 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
        'trade/min'         : 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
        'labels/ret'        : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
        'labels/ret_lag'    : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
        'benchmark/csi300'  : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI300' ,
        'benchmark/csi500'  : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI500' ,
        'benchmark/csi1000' : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI1000' ,
    }[source_key]

    source_dates = R_dir_dates(date_source)
    return np.array(sorted(source_dates) , dtype=int)
