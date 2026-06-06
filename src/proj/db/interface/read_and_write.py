"""Load, save, and path helpers for versioned tables under ``PATH.data`` (feather/parquet, tar, parallel IO)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any , Literal , Iterable , Callable , Union , TYPE_CHECKING

from src.proj.db.basic import dfHandler
from src.proj.db.io.dataframe import dfIOHandler , save_df , load_df , load_df_pl
from .db_path import DBPath

if TYPE_CHECKING:
    import polars as pl
    PL_MAPPER_TYPE = Union[Iterable[Callable[[pl.DataFrame], pl.DataFrame]] , Callable[[pl.DataFrame], pl.DataFrame] , None]
    PD_MAPPER_TYPE = Union[Iterable[Callable[[pd.DataFrame], pd.DataFrame]] , Callable[[pd.DataFrame], pd.DataFrame] , None]

__all__ = ['dfIOHandler' , 'save' , 'load' , 'loads' , 'load_pl' , 'loads_pl']

def save(df : pd.DataFrame | pl.DataFrame | None , db_src : str , db_key : str , date = None , *, 
         overwrite = True , indent = 1 , vb_level : Any = 1 , reason : str = ''):
    """
    Save data to database
    Parameters  
    ----------
    df: pd.DataFrame | None
        data to be saved
    db_src: str
        database source name , or export source name
    db_key: str
        database key , or export key name
    date: int, default None
        date to be saved, if the db is by date, date is required
    """
    if df is not None:
        df = dfHandler.reset_index_pandas(df , reset = True)
    db_path = DBPath(db_src , db_key)
    mark = save_df(
        df , db_path.path_exact(date) , overwrite = overwrite , 
        prefix = f'{db_src.title()} {reason}' if reason else db_key , 
        indent = indent , vb_level = vb_level)
    return mark

def load(db_src : str , db_key : str , date : int | None = None , *, 
         key_column = None , use_alt = False , closest = False , 
         missing_ok = True , indent = 1 , vb_level : Any = 1 , **kwargs) -> pd.DataFrame: 
    """
    Load data from database
    Parameters
    ----------
    db_src: str
        database source name , or export source name (etc. pred , factor , market_factor , factor_stats_daily , factor_stats_weekly)
    db_key: str
        database key , or export key name
    date: int, default None
        date to be loaded , if the db is by date , date is required
    key_column: str, default None
        date column name , if submitted , the date will be assigned to this column
    silent: default False
        if True, no message will be printed
    kwargs: kwargs for process_df
        missing_ok: bool, default True
            if True, return empty dataframe if the file does not exist
        ignored_fields: list, default []
            fields to be dropped , consider ['wind_id' , 'stockcode' , 'ticker' , 's_info_windcode' , 'code']
        reset_index: bool, default True
            if True, reset index (no drop index)
    """
    db_path = DBPath(db_src , db_key)
    df = load_df(db_path.path(date , use_alt = use_alt , closest = closest , indent = indent , vb_level = vb_level) , 
                 missing_ok = missing_ok , key_column = None)
    df = dfHandler.load_process_pandas(df , date , reassign_date_col = key_column , syntax = db_path.syntax(date) , indent = indent , vb_level = vb_level , **kwargs)
    return df

def load_pl(db_src : str , db_key : str , date : int | None = None , *, 
            key_column = None , use_alt = False , closest = False , 
            missing_ok = True , indent = 1 , vb_level : Any = 1 , **kwargs) -> pl.DataFrame: 
    """load dataframe from database but use polars to load"""
    db_path = DBPath(db_src , db_key)
    df = load_df_pl(db_path.path(date , use_alt = use_alt , closest = closest , indent = indent , vb_level = vb_level) , 
                    missing_ok = missing_ok , key_column = None)
    df = dfHandler.load_process_polars(df , date , reassign_date_col = key_column , syntax = db_path.syntax(date) , indent = indent , vb_level = vb_level , **kwargs)
    return df

def loads(db_src : str , db_key : str , dates : np.ndarray | list[int] | None = None , start : int | None = None , end : int | None = None , *,
          key_column = 'date' , override_existing_key = False , use_alt = False , closest = False ,
          accelerator : Literal['thread' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
          fill_datavendor = False , indent = 1 , vb_level : Any = 1 , **kwargs) -> pd.DataFrame:
    """load multiple dates from database"""
    assert DBPath.ByDate(db_src) , f'{db_src}.{db_key} is a name database, use load instead'
    db_path = DBPath(db_src , db_key)
    paths = db_path.get_paths(dates , start , end , use_alt = use_alt , closest = closest)
    if not paths:
        return pd.DataFrame()
    df = load_df(paths , key_column = key_column , override_existing_key = override_existing_key , accelerator = accelerator)
    df = dfHandler.load_process_pandas(df , syntax = db_path.syntax(dates) , indent = indent , vb_level = vb_level , **kwargs)
    if fill_datavendor:
        from src.data.loader import DATAVENDOR
        DATAVENDOR.db_loads_callback(df , db_src , db_key)
    return df

def loads_pl(db_src : str , db_key : str , dates : np.ndarray | list[int] | None = None , start : int | None = None , end : int | None = None , *,
             key_column : str | None = 'date' , override_existing_key = False , use_alt = False , closest = False ,
             accelerator : Literal['thread' , 'lazy'] | None = 'thread' , 
             fill_datavendor = False , indent = 1 , vb_level : Any = 1 , **kwargs) -> pl.DataFrame:
    """load multiple dates from database but use polars to load"""
    assert DBPath.ByDate(db_src) , f'{db_src}.{db_key} is a name database, use load_pl instead'
    import polars as pl
    db_path = DBPath(db_src , db_key)
    paths = db_path.get_paths(dates , start , end , use_alt = use_alt , closest = closest)
    if not paths:
        return pl.DataFrame()
    df = load_df_pl(paths , key_column = key_column , override_existing_key = override_existing_key , accelerator = accelerator)
    df = dfHandler.load_process_polars(df , syntax = db_path.syntax(dates) , indent = indent , vb_level = vb_level , **kwargs)
    if fill_datavendor:
        from src.data.loader import DATAVENDOR
        DATAVENDOR.db_loads_callback(df , db_src , db_key)
    return df
