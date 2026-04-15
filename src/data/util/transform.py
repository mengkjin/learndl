"""
Stateless helper functions for normalising and transforming DataFrames in the data pipeline.

Functions cover:
- Security identifier normalisation (``secid_adjust``)
- In-place column transformations (``col_reform``, ``row_filter``)
- Numeric precision reduction (``adjust_precision``)
- Minute-bar price filling and resampling (``trade_min_fillna``, ``trade_min_reform``)
- Chinese text → pinyin conversion (``chinese_to_pinyin``)
"""
import re
import pandas as pd
import numpy as np

from pypinyin import lazy_pinyin
from typing import Any
from src.proj import DB

def secid_adjust(df : pd.DataFrame ,
                 code_cols : str | list[str] = ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code' , 'symbol' , 'instrument' , 'ts_code' , 'stockid'] ,
                 drop_old = True , decode_first = False , raise_if_no_secid = True):
    """
    Normalise a variety of stock code columns into a single integer ``secid`` column.

    Scans ``df.columns`` for any name listed in ``code_cols`` (case-insensitive).
    If a matching column is found it is converted via ``DB.code2secid`` and renamed
    to ``secid``; the original column is dropped when ``drop_old=True``.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame with at least one code column.
    code_cols : str | list[str]
        Candidate column names to search for.  First match wins.
    drop_old : bool
        Rename the matched column to ``secid`` (True) or keep both (False).
    decode_first : bool
        Passed through to ``DB.code2secid`` for byte-string decoding.
    raise_if_no_secid : bool
        Raise ``ValueError`` when no recognised code column is found.
        If False the DataFrame is returned unchanged.
    """

    if isinstance(code_cols , str): 
        code_cols = [code_cols]
    code_cols = [col for col in df.columns if col.lower() in code_cols]
    if not code_cols: 
        code_col = 'secid'
    elif len(code_cols) > 1: 
        raise ValueError(f'Duplicated {code_cols} not supported')
    else:
        code_col = code_cols[0]
    if (code_col not in df.columns): 
        if raise_if_no_secid: 
            raise ValueError(f'secid column {code_col} not found in {df.columns}')
        else: 
            return df

    df[code_col]  = DB.code2secid(df[code_col] , decode_first = decode_first)
    if drop_old: 
        df = df.rename(columns={code_col:'secid'})
    else:
        df['secid'] = df[code_col]

    return df

def col_reform(df : pd.DataFrame , col : str , rename = None , fillna = None , astype = None , use_func = None):
    """
    Apply one or more in-place transformations to a single DataFrame column.

    Transformations are applied in this order: ``use_func`` → ``fillna`` → ``astype``.
    Pass ``rename`` to give the resulting column a new name.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to modify in-place.
    col : str
        Target column name.
    rename : str, optional
        New name for the column after transformation.
    fillna : scalar, optional
        Fill NaN values with this scalar before type conversion.
    astype : type, optional
        Cast the column to this dtype.
    use_func : callable, optional
        If provided, replaces fillna/astype — the column is set to ``use_func(df[col])``.
    """
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
    """
    Filter DataFrame rows by applying ``cond_func`` to one or more columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    col : str | list | tuple
        Column name(s) to evaluate.  When multiple columns are given they are
        passed as positional arguments to ``cond_func(*cols)``.
    cond_func : callable
        Returns a boolean Series/array used to select rows (default: identity).
    """
    if isinstance(col , str):
        return df[cond_func(df[col])]
    else:
        return df[cond_func(*[df[_c] for _c in col])]

def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
    """
    Downcast float columns to float32 and integer columns to int64.

    Near-zero float values (``|x| < tol``) are zeroed out to avoid
    precision noise propagating into downstream calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Frame modified in-place.
    tol : float
        Absolute threshold below which float values are set to 0.
    dtype_float : np.dtype
        Target dtype for floating-point columns (default: ``np.float32``).
    dtype_int : np.dtype
        Target dtype for integer columns (default: ``np.int64``).
    """
    for col in df.columns:
        if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
            df[col] = df[col].astype(dtype_float)
            df[col] *= (df[col].abs() > tol)
        if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
            df[col] = df[col].astype(dtype_int)
    return df

def trade_min_fillna(df : pd.DataFrame):
    """
    Fill missing values in minute-bar trade data.

    - ``amount`` and ``volume``: NaN → 0 (no trades occurred in that bar).
    - Price columns (``open``, ``high``, ``low``, ``vwap``, ``close``): NaN →
      the previous bar's close price (carry-forward within each security).

    The DataFrame must have columns ``secid``, ``minute``, ``open``, ``high``,
    ``low``, ``close``, ``amount``, ``volume``, and optionally ``vwap``.
    """
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
    """
    Resample minute-bar data from ``x_min_old``-minute bars to ``x_min_new``-minute bars.

    ``trade_min_fillna`` is applied first to ensure clean inputs.  Bars are
    grouped by ``(secid, new_minute_index)`` and aggregated with OHLCV rules:
    open=first, high=max, low=min, close=last, amount/volume=sum.
    VWAP is recomputed as ``amount / volume`` after aggregation.

    ``x_min_new`` must be a multiple of ``x_min_old``.
    """
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
        df = df[df['minute'] >= 0].copy()
    df['minute'] = df['minute'].clip(lower=0) // by
    agg_dict = {'open':'first','high':'max','low':'min','close':'last','amount':'sum','volume':'sum'}
    if 'num_trades' in df.columns: 
        agg_dict['num_trades'] = 'sum'
    data_new = df.groupby(['secid' , 'minute']).agg(agg_dict)
    assert isinstance(data_new , pd.DataFrame) , f'data_new is not a DataFrame: {data_new}'
    if 'vwap' in df.columns: 
        data_new['vwap'] = data_new['amount'] / data_new['volume']
    return data_new.reset_index(drop=False)

def chinese_to_pinyin(text : str):
    '''convert chinese characters to pinyin'''
    text = text.replace('\'' , '2').replace('因子' , '')
    hanzi_pattern = re.compile(r'[\u4e00-\u9fff]+')
    hanzi_parts = hanzi_pattern.findall(text)
    pinyin_parts = ['_'.join(lazy_pinyin(part)) for part in hanzi_parts]
    for hanzi, pinyin in zip(hanzi_parts, pinyin_parts):
        text = text.replace(hanzi, pinyin)
    return text