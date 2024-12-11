import numpy as np
import pandas as pd

from typing import Literal
from src.basic import CALENDAR

def code_to_secid(df : pd.DataFrame , code_col = 'ts_code' , retain = False , ashare = True):
    '''switch old symbol into secid'''
    if code_col not in df.columns.values: return df
    secid = df[code_col].astype(str).str.split('.').str[0].replace({'T00018' : '600018'})
    secid = secid.where(secid.str.isdigit() , '-1')
    if ashare: 
        market = df[code_col].astype(str).str.split('.').str[-1].str.lower()
        secid = secid.where(market.isin(['sh' , 'sz' , 'bj']) , -1)
    df['secid'] = secid.astype(int)
    if not retain: del df[code_col]
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

def quarter_ends(date , last_date = None , start_year = 1997 , consider_future = False , trailing_quarters = 3):
    date_list = np.sort(np.concatenate([np.arange(start_year , date // 10000 + 1) * 10000 + qe for qe in  [331,630,930,1231]]))
    if not consider_future: 
        date_list = date_list[date_list < date]
    
    if last_date is not None: 
        date_list_0 = date_list[date_list <= last_date][-trailing_quarters:] if trailing_quarters > 0 else []
        date_list_1 = date_list[date_list >  last_date]
        date_list = np.concatenate([date_list_0 , date_list_1])

    return date_list

def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
    '''adjust precision for df columns'''
    for col in df.columns:
        if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
            df[col] = df[col].astype(dtype_float)
            df[col] *= (df[col].abs() > tol)
        if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
            df[col] = df[col].astype(dtype_int)
    return df