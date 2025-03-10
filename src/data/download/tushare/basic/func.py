import numpy as np
import pandas as pd

from typing import Literal
from src.basic import CALENDAR , PATH
from src.data.util.basic.transform import secid_adjust

def ts_code_to_secid(df : pd.DataFrame , code_col = 'ts_code' , drop_old = True , ashare = True):
    '''switch old symbol into secid'''
    if code_col not in df.columns.values: return df
    if ashare: 
        valid = df[code_col].astype(str).str.split('.').str[-1].str.lower().isin(['sh' , 'sz' , 'bj'])
    else:
        valid = None
    df = secid_adjust(df , code_cols = code_col , drop_old = drop_old)
    if valid is not None: df['secid'] = df['secid'].where(valid , -1)
    return df

def updatable(date , last_date , freq : Literal['d' , 'w' , 'm']):
    if freq == 'd':
        return date > last_date
    elif freq == 'w':
        return date > CALENDAR.cd(last_date , 6)
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
    return np.sort(date_list)

def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
    '''adjust precision for df columns'''
    for col in df.columns:
        if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
            df[col] = df[col].astype(dtype_float)
            df[col] *= (df[col].abs() > tol)
        if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
            df[col] = df[col].astype(dtype_int)
    return df