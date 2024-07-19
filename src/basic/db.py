import re , os
import numpy as np
import pandas as pd

from typing import Any , Literal
from . import path as PATH 
from . import conf as CONF

DB_BY_NAME  : list[str] = ['information' , 'information_ts']
DB_BY_DATE  : list[str] = ['models' , 'trade' , 'labels' , 'benchmark' , 'membership_ts' , 'trade_ts']  

def list_files(directory , fullname = False , recur = False):
    '''list all files in directory'''
    if recur:
        paths = []
        for dirpath, _, filenames in os.walk(directory):
            paths += [os.path.join(dirpath , filename) for filename in filenames]
        if not fullname: paths = [os.path.relpath(p , directory) for p in paths]
    else:
        paths = os.listdir(directory)
        if fullname: paths = [os.path.join(directory , p) for p in paths]
    return paths

def get_target_path(db_src , db_key , date = None , makedir = False , 
                    force_type : Literal['name' , 'date'] | None = None):
    if db_src in DB_BY_NAME or force_type == 'name':
        db_path = os.path.join(PATH.database , f'DB_{db_src}')
        db_base = f'{db_key}.{CONF.SAVE_OPT_DB}'
    elif db_src in DB_BY_DATE or force_type == 'date':
        assert date is not None
        year_group = int(date) // 10000
        db_path = os.path.join(PATH.database , f'DB_{db_src}' , db_key , str(year_group))
        db_base = f'{db_key}.{str(date)}.{CONF.SAVE_OPT_DB}'
    else:
        raise KeyError(db_src)
    if makedir: os.makedirs(db_path , exist_ok=True)
    return os.path.join(db_path , db_base)

def get_source_dates(db_src , db_key):
    assert db_src in DB_BY_DATE
    return R_source_dates(db_src , db_key)

def get_target_dates(db_src , db_key , year = None):
    db_path = os.path.join(PATH.database , f'DB_{db_src}' , db_key)
    if year is None:
        target_files = list_files(db_path , recur=True)
    else:
        if not isinstance(year , list): year = [year]
        target_files = [f for y in year for f in list_files(f'{db_path}/{y}')]
    target_dates = R_path_date(target_files)
    return np.array(sorted(target_dates) , dtype=int)

def load_target_file(db_src , db_key , date = None) -> pd.DataFrame | Any:
    target_path = get_target_path(db_src , db_key , date)
    if os.path.exists(target_path):
        return load_df(target_path)
    else:
        return None

def save_df(df : pd.DataFrame , target_path):
    if CONF.SAVE_OPT_DB == 'feather':
        df.to_feather(target_path)
    else:
        df.to_parquet(target_path , engine='fastparquet')

def load_df(target_path):
    if CONF.SAVE_OPT_DB == 'feather':
        return pd.read_feather(target_path)
    else:
        return pd.read_parquet(target_path , engine='fastparquet')

def R_path_date(path , startswith = '' , endswith = '') -> list:
    '''get path date from R environment'''
    if isinstance(path , (list,tuple)):
        return [d[0] for d in [R_path_date(p , startswith , endswith) for p in path] if d]
    else:
        assert isinstance(path , str) , path
        if not path.startswith(startswith): return []
        if not path.endswith(endswith): return []
        s = os.path.basename(path).split('.')[-2][-8:]
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
