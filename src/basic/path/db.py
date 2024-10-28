import os , platform , re
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any , Literal

from . import glob as PATH 
from ...func.time import today

SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

DB_BY_NAME  : list[str] = ['information' , 'information_ts']
DB_BY_DATE  : list[str] = ['models' , 'trade' , 'labels' , 'benchmark' , 'membership_ts' , 'trade_ts' , 'financial_ts']  

def list_files(directory : str | Path , fullname = False , recur = False):
    '''list all files in directory'''
    if isinstance(directory , str): directory = Path(directory)
    if recur:
        paths : list[Path] = []
        #for dirpath, _, filenames in os.walk(directory):
        #    parent = Path(dirpath)
        #    paths += [parent.joinpath(filename) for filename in filenames]
        paths = [Path(dirpath).joinpath(filename) for dirpath, _, filenames in os.walk(directory) for filename in filenames]
    else:
        paths = [p for p in directory.iterdir()]
    paths = [p.absolute() for p in paths] if fullname else [p.relative_to(directory) for p in paths]
    return paths

def get_target_path(db_src , db_key , date = None , makedir = False , 
                    force_type : Literal['name' , 'date'] | None = None):
    if db_src in DB_BY_NAME or force_type == 'name':
        db_path = PATH.database.joinpath(f'DB_{db_src}')
        db_base = f'{db_key}.{SAVE_OPT_DB}'
    elif db_src in DB_BY_DATE or force_type == 'date':
        assert date is not None
        year_group = int(date) // 10000
        db_path = PATH.database.joinpath(f'DB_{db_src}' , db_key , str(year_group))
        db_base = f'{db_key}.{str(date)}.{SAVE_OPT_DB}'
    else:
        raise KeyError(db_src)
    if makedir: db_path.mkdir(exist_ok=True)
    return db_path.joinpath(db_base)

def get_source_dates(db_src , db_key):
    assert db_src in DB_BY_DATE
    return R_source_dates(db_src , db_key)

def get_target_dates(db_src , db_key , start_dt = None , end_dt = None , year = None):
    db_path = PATH.database.joinpath(f'DB_{db_src}' , db_key)
    if year is None:
        target_files = list_files(db_path , recur=True)
    else:
        if not isinstance(year , list): year = [year]
        target_files = [f for y in year for f in db_path.joinpath(str(y)).iterdir()]

    dates  = np.array(sorted(R_path_date(target_files)) , dtype=int)
    if end_dt   is not None: dates = dates[dates <= (end_dt   if end_dt   > 0 else today(end_dt))]
    if start_dt is not None: dates = dates[dates >= (start_dt if start_dt > 0 else today(start_dt))]
    
    return dates

def load_target_file(db_src , db_key , date = None) -> pd.DataFrame | Any:
    target_path = get_target_path(db_src , db_key , date)
    if target_path.exists():
        df = load_df(target_path)
        if df.empty:
            print(f'{db_src} , {db_key} , {date} entry is empty')
        elif df.isna().all().all():
            print(f'{db_src} , {db_key} , {date} entry is all-NA')
        elif df.isna().all().any():
            cols = df.columns.values[df.isna().all()]
            print(f'{db_src} , {db_key} , {date} entry has columns [{str(cols)}] all-NA')
        return df
    else:
        return None

def load_target_file_dates(db_src , db_key , dates , 
                           parallel : Literal['thread' , 'process'] | None = 'thread' , 
                           max_workers = 20) -> dict[int,pd.DataFrame|None]:
    if parallel is None:
        dfs = {date:load_target_file(db_src , db_key , date) for date in dates}
    else:
        assert parallel == 'thread' or platform.system() != 'Windows' , (parallel , platform.system())
        if n_cpu:= os.cpu_count(): max_workers = min(max_workers , n_cpu)
        PoolExecutor = ThreadPoolExecutor if parallel == 'thread' else ProcessPoolExecutor
        with PoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(load_target_file, db_src , db_key , date):date for date in dates}
            dfs = {futures[future]:future.result() for future in as_completed(futures)}
    return dfs
 
def save_df(df : pd.DataFrame , target_path):
    if SAVE_OPT_DB == 'feather':
        df.to_feather(target_path)
    else:
        df.to_parquet(target_path , engine='fastparquet')

def load_df(target_path):
    if SAVE_OPT_DB == 'feather':
        return pd.read_feather(target_path)
    else:
        return pd.read_parquet(target_path , engine='fastparquet')

def R_path_date(path : Path | list[Path] | tuple[Path] , startswith = '' , endswith = '') -> list:
    '''get path date from R environment'''
    if isinstance(path , (list,tuple)):
        return [d[0] for d in [R_path_date(p , startswith , endswith) for p in path] if d]
    else:
        if not path.name.startswith(startswith): return []
        if not path.name.endswith(endswith): return []
        s = path.stem[-8:]
        return [int(s)] if s.isdigit() else []
    
def R_dir_dates(directory):
    '''get all path dates in a dir from R environment'''
    return R_path_date(list_files(directory , recur = True))
    
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
