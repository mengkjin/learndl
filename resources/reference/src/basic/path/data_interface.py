import os , platform , re
import dask
import dask.dataframe
from dask.delayed import delayed
from dask.base import compute
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal , Callable

from src.project_setting import MACHINE

from . import path_structure as PATH
from .code_mapper import secid_to_secid

SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

DB_BY_NAME  : list[str] = ['information_js' , 'information_ts']
DB_BY_DATE  : list[str] = ['models' , 'sellside' ,
                           'trade_js' , 'labels_js' , 'benchmark_js' , 
                           'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'holding_ts'
                           ]  

DB_ALTERNATIVES : dict[str , str] = {
    'trade_ts' : 'trade_js' ,
    'benchmark_ts' : 'benchmark_js'
}
LOAD_PARALLEL : Literal['thread' , 'process' , 'dask' , 'none'] = 'thread'
# in Mac-Matthew , thread > dask > none > process

LOAD_MAX_WORKERS : int | Any = 40 if MACHINE.server else os.cpu_count()

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
        if MACHINE.server:
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
    paths = filter_paths(paths)
    return paths

def filter_paths(paths : list[Path] , ignore_prefix = ('.' , '~')):
    return [p for p in paths if not p.name.startswith(ignore_prefix)]

def paths_to_dates(paths : list[Path]):
    dates = [int(p.stem[-8:]) for p in paths if p.stem[-8:].isdigit()]
    dates = np.array(sorted(dates) , dtype=int)
    return dates

def dir_dates(directory : Path , start_dt = None , end_dt = None , year = None):
    paths = [Path(file) for _ , _ , files in os.walk(directory) for file in files]
    dates = paths_to_dates(paths)
    if end_dt   is not None: dates = dates[dates <= (end_dt   if end_dt   > 0 else today(end_dt))]
    if start_dt is not None: dates = dates[dates >= (start_dt if start_dt > 0 else today(start_dt))]
    if year is not None:     dates = dates[dates // 10000 == year]
    return dates

def dir_min_date(directory : Path):
    years = [int(y.stem) for y in directory.iterdir() if y.is_dir()]
    if not years: return 0
    paths = [p for p in directory.joinpath(str(min(years))).iterdir()]
    dates = paths_to_dates(paths)
    return min(dates) if len(dates) else 0

def dir_max_date(directory : Path):
    years = [int(y.stem) for y in directory.iterdir() if y.is_dir()]
    if not years: return 99991231
    paths = [p for p in directory.joinpath(str(max(years))).iterdir()]
    dates = paths_to_dates(paths)
    return max(dates) if len(dates) else 99991231

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
        if printing_prefix: print(f'{printing_prefix} save to {path} successfully')
        return True
    else:
        if printing_prefix: print(f'{printing_prefix} already exists')
        return False

def load_df(path : Path , raise_if_not_exist = False):
    if not path.exists():
        if raise_if_not_exist: raise FileNotFoundError(path)
        else: return pd.DataFrame()
    if SAVE_OPT_DB == 'feather':
        df = pd.read_feather(path)
    else:
        df = pd.read_parquet(path)
    df = load_df_mapper(df)
    return df

def load_df_mapper(df : pd.DataFrame):
    old_index = df.index.names if 'secid' in df.index.names else None
    if old_index is not None: df = df.reset_index(drop = False)
    if 'secid' in df.columns:  df['secid'] = secid_to_secid(df['secid'])
    if old_index is not None: df = df.set_index(old_index)
    return df

def load_df_multi(paths : dict , date_colname : str = 'date' , 
                  parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread'):
    if parallel is None: parallel = LOAD_PARALLEL
    reader : Any = pd.read_feather if SAVE_OPT_DB == 'feather' else pd.read_parquet
    paths = {d:p for d,p in paths.items() if p.exists()}
    if parallel is None or parallel == 'none':
        dfs = [reader(p).assign(**{date_colname:d}) for d,p in paths.items()]
    elif parallel == 'dask':
        #ddf = dask.dataframe.from_delayed([delayed(reader)(p).assign(**{date_colname:d}) for d,p in paths.items()])
        #df = ddf.compute()
        #return load_df_mapper(df)
        ddfs = [delayed(reader)(p).assign(**{date_colname:d}) for d,p in paths.items()]
        dfs = compute(ddfs)[0]
    else:
        assert parallel == 'thread' or platform.system() != 'Windows' , (parallel , platform.system())
        max_workers = min(LOAD_MAX_WORKERS , max(len(paths) // 5 , 1))
        PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(reader , p):d for d,p in paths.items()}
            dfs = [future.result().assign(**{date_colname:futures[future]}) for future in as_completed(futures)]

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs)
    df = load_df_mapper(df)
    return df

def process_df(df : pd.DataFrame , date = None, date_colname = None , check_na_cols = False , 
               df_syntax : str | None = 'some df' , reset_index = True , ignored_fields = []):
    if date_colname and date is not None: df[date_colname] = date

    if df_syntax:
        if df.empty:
            print(f'{df_syntax} is empty')
        elif df.isna().all().all():
            print(f'{df_syntax} is all-NA')
        elif check_na_cols and df.isna().all().any():
            print(f'{df_syntax} has columns [{str(df.columns.values[df.isna().all()])}] all-NA')

    if reset_index and len(df.index.names) > 1 or df.index.name: df = df.reset_index()
    if ignored_fields: df = df.drop(columns=ignored_fields , errors='ignore')
    return df

def db_path(db_src , db_key , date = None , use_alt = False) -> Path:
    if db_src in DB_BY_NAME:
        parent = PATH.database.joinpath(f'DB_{db_src}')
        base = f'{db_key}.{SAVE_OPT_DB}'
    elif db_src in DB_BY_DATE:
        assert date is not None
        parent = PATH.database.joinpath(f'DB_{db_src}' , db_key , str(int(date) // 10000))
        base = f'{db_key}.{str(date)}.{SAVE_OPT_DB}'
    else:
        raise KeyError(db_src)
    new_path = parent.joinpath(base)
    if not new_path.exists() and db_src in DB_ALTERNATIVES and use_alt:
        alt_path = db_path(DB_ALTERNATIVES[db_src] , db_key , date , use_alt = False)
        if alt_path.exists(): new_path = alt_path
    return new_path

def db_dates(db_src , db_key , start_dt = None , end_dt = None , year = None , use_alt = False):
    path = PATH.database.joinpath(f'DB_{db_src}' , db_key)
    dates = dir_dates(path , start_dt , end_dt , year)
    if db_src in DB_ALTERNATIVES and use_alt:
        alt_src   = DB_ALTERNATIVES[db_src]
        alt_path  = PATH.database.joinpath(f'DB_{alt_src}' , db_key)
        alt_dates = np.setdiff1d(dir_dates(alt_path , start_dt , end_dt , year) , dates)
        dates = np.concatenate([alt_dates , dates])
    return dates

def db_min_date(db_src , db_key):
    return min(db_dates(db_src , db_key))

def db_max_date(db_src , db_key):
    return max(db_dates(db_src , db_key))

# @db_src_deprecated(1)
def db_save(df : pd.DataFrame | None , db_src , db_key , date = None , verbose = True):
    '''
    Save data to database
    Parameters  
    ----------
    df: pd.DataFrame | None
        data to be saved
    db_src: str
        database source name
    db_key: str
        database key
    date: int, default None
        date to be saved, if the db is by date, date is required
    '''
    printing_prefix = f'DataBase object [{db_src}],[{db_key}],[{date}]' if verbose else None
    if df is not None and (len(df.index.names) > 1 or df.index.name): df = df.reset_index()
    mark = save_df(df , db_path(db_src , db_key , date , use_alt = False) , 
                   overwrite = True , printing_prefix = printing_prefix)
    return mark

# @db_src_deprecated(0)
def db_load(db_src , db_key , date = None , date_colname = None , verbose = True , use_alt = False , 
            raise_if_not_exist = False , **kwargs) -> pd.DataFrame: 
    '''
    Load data from database
    Parameters
    ----------
    db_src: str
        database source name : ['information_js' , 'information_ts' , 'trade_js' , 'labels_js' , 'benchmark_js' , 
                               'trade_ts' , 'financial_ts' , 'analyst_ts' , 'labels_ts' , 'benchmark_ts' , 'membership_ts' , 'sellside']
    db_key: str
        database key that can be found in the database directory
    date: int, default None
        date to be loaded , if the db is by date , date is required
    date_colname: str, default None
        date column name , if submitted , the date will be assigned to this column
    silent: default False
        if True, no message will be printed
    kwargs: kwargs for process_df
        raise_if_not_exist: bool, default False
            if True, raise FileNotFoundError if the file does not exist
        ignored_fields: list, default []
            fields to be dropped , consider ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code']
        reset_index: bool, default True
            if True, reset index (no drop index)
    '''
    path = db_path(db_src , db_key , date , use_alt = use_alt)
    df_syntax = f'{db_src}/{db_key}/{date}' if verbose else None
    df = load_df(path , raise_if_not_exist = raise_if_not_exist)
    df = process_df(df , date , date_colname , df_syntax = df_syntax , **kwargs)
    return df

# @db_src_deprecated(0)
def db_load_multi(db_src , db_key , dates = None , start_dt = None , end_dt = None , 
                  date_colname = 'date' , 
                  verbose = True , use_alt = False , 
                  parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , **kwargs):
    if dates is None:
        assert start_dt is not None and end_dt is not None , f'start_dt and end_dt must be provided if dates is not provided'
        dates = db_dates(db_src , db_key , start_dt , end_dt , use_alt = use_alt)
    paths : dict[int , Path] = {int(date):db_path(db_src , db_key , date , use_alt = use_alt) for date in dates}
    df_syntax = f'{db_src}/{db_key}/multi-dates' if verbose else None
    df = load_df_multi(paths , date_colname , parallel)
    df = process_df(df , df_syntax = df_syntax , **kwargs)
    return df

def db_rename(db_src , db_key , new_db_key):
    assert new_db_key not in list_files(PATH.database.joinpath(f'DB_{db_src}' , db_key)) , f'{new_db_key} already exists'
    if db_src in DB_BY_NAME:
        old_path = db_path(db_src , db_key)
        new_path = db_path(db_src , new_db_key)
        old_path.rename(new_path)
    else:
        for date in db_dates(db_src , db_key):
            old_path = db_path(db_src , db_key , date)
            new_path = db_path(db_src , new_db_key , date)
            new_path.parent.mkdir(parents=True , exist_ok=True)
            old_path.rename(new_path)
        root = PATH.database.joinpath(f'DB_{db_src}' , db_key)
        [d.rmdir() for d in root.iterdir() if d.is_dir()]
        root.rmdir()


def pred_path(model_name : str , date : int | Any):
    return PATH.preds.joinpath(model_name , str(date // 10000) , f'{model_name}.{date}.feather')

def pred_dates(model_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.preds.joinpath(model_name) , start_dt , end_dt , year)

def pred_min_date(model_name : str):
    return dir_min_date(PATH.preds.joinpath(model_name))

def pred_max_date(model_name : str):
    return dir_max_date(PATH.preds.joinpath(model_name))

def pred_save(df : pd.DataFrame | None , model_name : str , date : int | Any , overwrite = True):
    return save_df(df , pred_path(model_name , date) , overwrite)

def pred_load(model_name : str , date : int | Any , date_colname = None , verbose = True , **kwargs):
    df = load_df(pred_path(model_name , date))
    df_syntax = f'pred/{model_name}/{date}' if verbose else None
    return process_df(df , date , date_colname , df_syntax = df_syntax , **kwargs)

def pred_load_multi(model_name : str , dates = None , start_dt = None , end_dt = None , 
                    date_colname = 'date' , verbose = True , 
                    parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , **kwargs):
    if dates is None:
        assert start_dt is not None and end_dt is not None , f'start_dt and end_dt must be provided if dates is not provided'
        dates = pred_dates(model_name , start_dt , end_dt)
    paths = {date:pred_path(model_name , date) for date in dates}
    df = load_df_multi(paths , date_colname , parallel)
    df_syntax = f'pred/{model_name}/multi-dates' if verbose else None
    return process_df(df , df_syntax = df_syntax , **kwargs)

def factor_path(factor_name : str , date : int | Any):
    return PATH.factor.joinpath(factor_name , str(date // 10000) , f'{factor_name}.{date}.feather')

def factor_dates(factor_name : str , start_dt = None , end_dt = None , year = None):
    return dir_dates(PATH.factor.joinpath(factor_name) , start_dt , end_dt , year)

def factor_min_date(factor_name : str):
    return dir_min_date(PATH.factor.joinpath(factor_name))

def factor_max_date(factor_name : str):
    return dir_max_date(PATH.factor.joinpath(factor_name))

def factor_save(df : pd.DataFrame | None , factor_name : str , date : int | Any , overwrite = True):
    return save_df(df , factor_path(factor_name , date) , overwrite)

def factor_load(factor_name : str , date : int | Any , date_colname = None , verbose = True , **kwargs):
    df = load_df(factor_path(factor_name , date))
    df_syntax = f'factor/{factor_name}/{date}' if verbose else None
    return process_df(df , date , date_colname , df_syntax = df_syntax , **kwargs)

def factor_load_multi(factor_name : str , dates = None , start_dt = None , end_dt = None , 
                      date_colname = 'date' , verbose = True , 
                      parallel : Literal['thread' , 'process' , 'dask' , 'none'] | None = 'thread' , **kwargs):
    if dates is None:
        assert start_dt is not None and end_dt is not None , f'start_dt and end_dt must be provided if dates is not provided'
        dates = factor_dates(factor_name , start_dt , end_dt)
    paths = {date:factor_path(factor_name , date) for date in dates}
    df = load_df_multi(paths , date_colname , parallel)
    df_syntax = f'factor/{factor_name}/multi-dates' if verbose else None
    return process_df(df , df_syntax = df_syntax , **kwargs)

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
        'benchmark_js/csi800'  : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI800' ,
        'benchmark_js/csi1000' : 'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/CSI1000' ,
    }[source_key]

    source_dates = R_dir_dates(date_source)
    return np.array(sorted(source_dates) , dtype=int)
