import pandas as pd
import numpy as np

from typing import Any
from src.proj import DB

def secid_adjust(df : pd.DataFrame , code_cols : str | list[str] = ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code'] , 
                 drop_old = True , decode_first = False , raise_if_no_secid = True):
    '''switch various type of codes into secid'''

    if isinstance(code_cols , str): 
        code_cols = [code_cols]
    code_cols = [col for col in df.columns.values if str(col).lower() in code_cols]
    if not code_cols: 
        code_cols = ['secid']
    elif len(code_cols) > 1: 
        raise ValueError(f'Duplicated {code_cols} not supported')
    code_cols = code_cols[0]
    if (code_cols not in df.columns): 
        if raise_if_no_secid: 
            raise ValueError(f'secid not found')
        else: 
            return df

    df['secid'] = DB.code_to_secid(df[code_cols] , decode_first = decode_first)
    if drop_old and (code_cols != 'secid'): 
        df = df.drop(columns=[code_cols])

    return df

def col_reform(df : pd.DataFrame , col : str , rename = None , fillna = None , astype = None , use_func = None):
    '''do certain processing to DataFrame columns: newcol(rename) , fillna , astype or use_func'''
    if use_func is not None:
        df[col] = use_func(df[col])
    else:
        x = df[col]
        if fillna is not None: 
            x = x.fillna(fillna)
        if astype is not None: 
            x = x.astype(astype)
        df[col] = x 
    if rename: 
        df = df.rename(columns={col:rename})
    return df

def row_filter(df : pd.DataFrame | Any , col : str | list | tuple , cond_func = lambda x:x) -> pd.DataFrame | Any:
    '''filter pd.DataFrame rows: cond_func(col)'''
    if isinstance(col , str):
        return df[cond_func(df[col])]
    else:
        return df[cond_func(*[df[_c] for _c in col])]

def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
    '''adjust precision for df columns'''
    for col in df.columns:
        if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
            df[col] = df[col].astype(dtype_float)
            df[col] *= (df[col].abs() > tol)
        if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
            df[col] = df[col].astype(dtype_int)
    return df

def trade_min_fillna(df : pd.DataFrame):
    '''fillna for minute trade data'''
    #'amount' , 'volume' to 0
    df['amount'] = df['amount'].where(~df['amount'].isna() , 0)
    df['volume'] = df['volume'].where(~df['volume'].isna() , 0)

    # close price
    df1 = df.loc[:,['secid','minute','close']].copy().rename(columns={'close':'fix'})
    df1['minute'] = df1['minute'] - df1['minute'].min()
    df = df.merge(df1,on=['secid','minute'],how='left')
    df['fix'] = df['fix'].where(~df['fix'].isna() , df['open'])

    # prices to last time price (min != 0)
    for feat in ['open','high','low','vwap','close']: 
        df[feat] = df[feat].where(~df[feat].isna() , df['fix'])
    df = df.drop(columns=['fix'])
    return df

def trade_min_reform(df : pd.DataFrame , x_min_new : int , x_min_old = 1):
    '''from minute trade data to xmin trade data'''
    df = trade_min_fillna(df)
    if x_min_new == x_min_old: 
        return df
    assert x_min_new % x_min_old == 0 , f'{x_min_new} % {x_min_old} != 0'
    by = x_min_new // x_min_old
    max_nbar = 240 // x_min_old
    assert df['minute'].max() in [max_nbar-1 , max_nbar] , f'{df["minute"].max()} should be {max_nbar-1} or {max_nbar}'
    if df['minute'].max() == max_nbar: 
        df['minute'] = df['minute'] -1
    if x_min_new != 1: 
        df = df.query('minute >= 0')
    df['minute'] = df['minute'].clip(lower=0) // by
    agg_dict = {'open':'first','high':'max','low':'min','close':'last','amount':'sum','volume':'sum'}
    if 'num_trades' in df.columns: 
        agg_dict['num_trades'] = 'sum'
    data_new = df.groupby(['secid' , 'minute']).agg(agg_dict)
    assert isinstance(data_new , pd.DataFrame) , f'data_new is not a DataFrame: {data_new}'
    if 'vwap' in df.columns: 
        data_new['vwap'] = data_new['amount'] / data_new['volume']
    return data_new.reset_index(drop=False)