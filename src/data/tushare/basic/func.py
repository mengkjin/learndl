# 导入tushare
import os , time
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any , Literal

from ....basic import get_target_path
from ....func import date_diff , today

def code_to_secid(df : pd.DataFrame , code_col = 'ts_code' , retain = False):
    '''switch old symbol into secid'''
    if code_col not in df.columns.values: return df
    df['secid'] = df[code_col].astype(str).str.slice(0, 6).replace({'T00018' : '600018'})
    df['secid'] = df['secid'].where(df['secid'].str.isdigit() , '-1').astype(int)
    if not retain: del df[code_col]
    return df

def updatable(date , last_date , freq : Literal['d' , 'w' , 'm']):
    if freq == 'd':
        return date > last_date
    elif freq == 'w':
        return date_diff(date , last_date) > 6
    elif freq == 'm':
        return ((date // 100) % 100) != ((last_date // 100) % 100)
    
def dates_to_update(date , last_date , freq : Literal['d' , 'w' , 'm']):
    if last_date >= date: return np.array([])
    if freq == 'd':
        date_list = pd.date_range(str(last_date) , str(date)).strftime('%Y%m%d').astype(int).to_numpy()[1:]
    elif freq == 'w':
        date_list = pd.date_range(str(last_date) , str(date)).strftime('%Y%m%d').astype(int).to_numpy()[::7][1:]
    elif freq == 'm':
        date_list = pd.date_range(str(last_date) , str(date) , freq='ME').strftime('%Y%m%d').astype(int).to_numpy()
        if last_date in date_list: date_list = date_list[1:]
    return date_list

def quarter_ends(date , last_date = None , start_year = 1997):
    date_list = np.sort(np.concatenate([np.arange(start_year , date // 10000 + 1) * 10000 + qe for qe in  [331,630,930,1231]]))
    date_list = date_list[date_list < date]
    
    if last_date is not None: 
        date_list_0 = date_list[date_list <= last_date][-3:]
        date_list_1 = date_list[date_list >  last_date]
        date_list = np.concatenate([date_list_0 , date_list_1])

    return date_list

def complete_calendar():
    cal = pd.read_feather(get_target_path('information_ts' , 'calendar'))
    trd = cal[cal['trade'] == 1].reset_index(drop=True)
    trd['pre'] = trd['calendar'].shift(1, fill_value=-1)
    return trd.reset_index()

def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
    '''adjust precision for df columns'''
    for col in df.columns:
        if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
            df[col] = df[col].astype(dtype_float)
            df[col] *= (df[col].abs() > tol)
        if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
            df[col] = df[col].astype(dtype_int)
    return df

def file_update_date(path : Path , default = 19970101):
    if path.exists():
        return int(time.strftime('%Y%m%d',time.localtime(os.path.getmtime(path))))
    else:
        return default