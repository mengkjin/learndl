"""Load, save, and path helpers for versioned tables under ``PATH.data`` (feather/parquet, tar, parallel IO)."""
from __future__ import annotations

from dask.delayed import delayed
from dask.base import compute
import numpy as np
import pandas as pd
import polars as pl
import io

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any , Literal , Mapping

from src.proj.env import MACHINE
from src.proj.log import Logger

from .core import DATAFRAME_SUFFIX , PATH_TYPE , PATHS_TYPE , PD_MAPPER_TYPE , PL_MAPPER_TYPE
from .df_handler import dfHandler
from .db_path import path_date , DBPath

__all__ = [
    'dfIOHandler' ,
    'save' , 'save_df' , 'append_df' , 
    'load' , 'loads' , 'load_pl' , 'loads_pl' , 
    'load_df' , 'load_dfs' , 'load_df_pl' , 'load_dfs_pl' , 
    'load_df_max_date' , 'load_df_min_date'
]

class dfIOHandler:
    """File IO operations handler"""
    @classmethod
    def load_pandas(
        cls , path : PATH_TYPE | io.BytesIO , * , 
        missing_ok = True , 
        mapper : PD_MAPPER_TYPE = None
    ) -> pd.DataFrame:
        """load dataframe from path"""
        if isinstance(path , PATH_TYPE) and not Path(path).exists() and missing_ok: 
            return pd.DataFrame()
        if DATAFRAME_SUFFIX == 'feather':
            df = pd.read_feather(path)
        else:
            df = pd.read_parquet(path , engine='fastparquet')
        df = dfHandler.apply_mapper(df , mapper)
        return df

    @classmethod
    def load_polars(cls , path : PATH_TYPE | io.BytesIO , * , missing_ok = True , mapper : PL_MAPPER_TYPE = None) -> pl.DataFrame:
        """load dataframe from path"""
        if isinstance(path , PATH_TYPE) and not Path(path).exists() and missing_ok: 
            return pl.DataFrame()
        if DATAFRAME_SUFFIX == 'feather':
            df = pl.read_ipc(path , memory_map = False)
        else:
            df = pl.read_parquet(path , memory_map = False)
        df = dfHandler.apply_mapper(df , mapper)
        return df
    
    @classmethod
    def save_df(cls , df : pd.DataFrame | pl.DataFrame , path : PATH_TYPE | io.BytesIO):
        """save dataframe to path"""
        if isinstance(df , pd.DataFrame) and None in df.index.names:
            df = df.reset_index(None , drop = True)
        try:
            if isinstance(df , pd.DataFrame) and DATAFRAME_SUFFIX == 'feather':
                df.to_feather(path)
            elif isinstance(df , pd.DataFrame) and DATAFRAME_SUFFIX == 'parquet':
                df.to_parquet(path , engine='fastparquet')
            elif isinstance(df , pl.DataFrame) and DATAFRAME_SUFFIX == 'feather':
                df.write_ipc(path)
            elif isinstance(df , pl.DataFrame) and DATAFRAME_SUFFIX == 'parquet':
                df.write_parquet(path)
            else:
                raise ValueError(f'Unsupported dataframe type {type(df)} with suffix {DATAFRAME_SUFFIX}')
        except Exception as e:
            Logger.error(f'Error saving {path}: {e}')
            Logger.display(df , caption = 'Error saving DataFrame')
            raise

    @classmethod
    def to_path_dict(cls , paths : PATHS_TYPE) -> dict[int | Any, Path]:
        """convert paths list or dict to path dict"""
        if isinstance(paths , Mapping):
            return {key:Path(path) for key,path in paths.items()}
        else:
            try:
                return {path_date(p):Path(p) for p in paths}
            except Exception:
                return {i:Path(p) for i,p in enumerate(paths)}

    @classmethod
    def load_pandas_multiple(
        cls , paths : PATHS_TYPE , * ,
        accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
        mapper : PD_MAPPER_TYPE = None
    ) -> dict[int | Any, pd.DataFrame]:
        """load dataframe from multiple paths in accelerating mode"""
        paths = {d:p for d,p in cls.to_path_dict(paths).items() if p.exists()}
        if not paths:
            return {}

        if accelerator in ['polars' , 'polars_thread']:
            polars_accelerator = 'thread' if accelerator == 'polars_thread' else None
            dfs = cls.load_polars_multiple(paths , accelerator = polars_accelerator , mapper = None)
            dfs = {d:dfHandler.apply_mapper(df.to_pandas() , mapper) for d,df in dfs.items()}
            return dfs

        def loader(p : PATH_TYPE) -> pd.DataFrame:
            return cls.load_pandas(p , mapper = mapper)
        if accelerator is None:
            dfs = {d:loader(p) for d,p in paths.items() if not loader(p).empty}
        elif accelerator == 'dask':
            ddfs = [delayed(loader)(p) for d,p in paths.items()]
            dfs = {d:df for d,df in zip(paths.keys() , compute(ddfs)[0])}
        elif accelerator in ['thread' , 'process']:
            assert accelerator == 'thread' or (not MACHINE.is_windows and accelerator == 'process'), (accelerator , MACHINE.system_name)
            max_workers = min(MACHINE.max_workers , max(len(paths) // 5 , 1))
            PoolExecutor = ThreadPoolExecutor if accelerator == 'thread' else ProcessPoolExecutor
            with PoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(loader , p):d for d,p in paths.items()}
                dfs = {futures[future]:future.result() for future in as_completed(futures)}
        else:
            raise ValueError(f'Unsupported accelerator: {accelerator}')
        return dfs

    @classmethod
    def load_polars_multiple(
        cls , paths : PATHS_TYPE , * , 
        accelerator : Literal['thread'] | None = 'thread' , 
        mapper : PL_MAPPER_TYPE = None ,
    ) -> dict[int | Any, pl.DataFrame]:
        """
        load dataframe from multiple paths in accelerating mode
        """
        def loader(p : PATH_TYPE) -> pl.DataFrame:
            return cls.load_polars(p , mapper = mapper)

        paths = {d:p for d,p in cls.to_path_dict(paths).items() if p.exists()}
        if not paths:
            return {}
        if accelerator is None:
            dfs = {d:loader(p) for d,p in paths.items()}
        elif accelerator == 'thread':
            max_workers = min(MACHINE.max_workers , max(len(paths) // 5 , 1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(loader , p):d for d,p in paths.items()}
                dfs = {futures[future]:future.result() for future in as_completed(futures)}
        else:
            raise ValueError(f'Unsupported accelerator: {accelerator}')
        return dfs
  
def save_df(df : pd.DataFrame | None , path : PATH_TYPE , *, overwrite = True , prefix = '' , empty_ok = False , indent = 1 , vb_level : Any = 1):
    """save dataframe to path"""
    if df is None or (not empty_ok and df.empty): 
        return False
    prefix = prefix or ''
    path = Path(path)
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True , exist_ok=True)
        dfIOHandler.save_df(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def append_df(df : pd.DataFrame | None , path : PATH_TYPE , *, drop_duplicate_cols : list[str] | None = None , prefix = '' , indent = 1 , vb_level : Any = 1):
    """append dataframe to path , can pass drop_duplicate_cols to drop duplicate columns"""
    if df is None or df.empty: 
        return False

    path = Path(path)
    if not path.exists():
        return save_df(df , path , overwrite = True , prefix = prefix , indent = indent , vb_level = vb_level)
    else:
        status = 'Appended'
        df = pd.concat([load_df(path) , df])
        if drop_duplicate_cols:
            df = df.drop_duplicates(subset=drop_duplicate_cols , keep='last')
            status += f'with unique ({",".join(drop_duplicate_cols)})'
        dfIOHandler.save_df(df , path)
        Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)

def load_df(
    path : PATH_TYPE | PATHS_TYPE , * , 
    missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
    accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
    mapper : PD_MAPPER_TYPE = None
):
    """
    load dataframe from path or paths
    Parameters
    ----------
    path : Path | str | Iterable[Path | str] | dict[int, Path | str]
        path or paths to load , key is date
    missing_ok : bool
        if True, return empty dataframe for missing path(s)
    key_column : str | None
        key column name , if None, use date column
    accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread']
        accelerating mode
    mapper : Callable[[pd.DataFrame], pd.DataFrame]
        mapper function to execute on each dataframe
    """
    if isinstance(path , PATH_TYPE):
        df = dfIOHandler.load_pandas(path , missing_ok = missing_ok , mapper = None)
    
    else:
        if accelerator in ['polars' , 'polars_thread']:
            polars_accelerator = 'thread' if accelerator == 'polars_thread' else None
            df = load_df_pl(path , missing_ok = missing_ok , key_column = key_column , override_existing_key = override_existing_key , accelerator = polars_accelerator , mapper = None).to_pandas()
        else:
            dfs = dfIOHandler.load_pandas_multiple(path , accelerator = accelerator , mapper = None)
            if not dfs and missing_ok:
                return pd.DataFrame()
            temp_key = f'_concat_index_{np.random.randint(1000000)}'
            df = pd.concat(dfs , names = [temp_key])
            if key_column is None:
                df = df.reset_index([temp_key] , drop = True)
            elif (key_column in df.columns or key_column in df.index.names) and not override_existing_key:
                Logger.alert1(f'key_column {key_column} already exists in dataframe columns [{df.columns}] or '
                                f'index names [{df.index.names}] , if you want to override it, set override_existing_key to True')
                df = df.reset_index([temp_key] , drop = True)
            else:
                if key_column in df.index.names:
                    df = df.reset_index([key_column] , drop = True)
                df = df.drop(columns = [key_column], errors='ignore').\
                    reset_index([temp_key] , drop = False).rename(columns = {temp_key:key_column})
    return dfHandler.wrapped_mapper(mapper)(df)

def load_dfs(
    paths : PATHS_TYPE , * ,  
    accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
    mapper : PD_MAPPER_TYPE = None
) -> dict[int | Any, pd.DataFrame]:
    """
    load dataframe from multiple paths , return dict of date and dataframe
    Parameters
    ----------
    paths : dict[int, Path | str] | Iterable[Path | str]
        paths to load , key is date
    accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread'] | None
        accelerating mode
    mapper : Iterable[Callable[[pd.DataFrame], pd.DataFrame]] | Callable[[pd.DataFrame], pd.DataFrame] | None
        mapper function to execute on each dataframe
    """
    return dfIOHandler.load_pandas_multiple(paths , accelerator = accelerator , mapper = dfHandler.wrapped_mapper(mapper))

def load_df_pl(
    path : PATH_TYPE | PATHS_TYPE , *, 
    missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
    accelerator : Literal['thread' , 'lazy'] | None = 'thread' , 
    mapper : PL_MAPPER_TYPE = None
) -> pl.DataFrame:
    """
    load polars dataframe from path or paths
    Parameters
    ----------
    path : Path | str | Iterable[Path | str] | dict[int, Path | str]
        path or paths to load , key is date
    missing_ok : bool
        if True, return empty dataframe for missing path(s)
    accelerator : Literal['thread' , 'lazy'] | None
        accelerating mode
    mapper : Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pl.DataFrame] | None
        mapper function to execute on each dataframe
    """
    if isinstance(path , PATH_TYPE):
        df = dfIOHandler.load_polars(path , missing_ok = missing_ok , mapper = None)
    else:
        if accelerator == 'lazy':
            path_dict = dfIOHandler.to_path_dict(path)
            dfs = {key:pl.scan_ipc(path) for key,path in path_dict.items() if path.exists()}
        else:
            dfs = dfIOHandler.load_polars_multiple(path , accelerator = accelerator , mapper = None)
            
        if not dfs and missing_ok:
            return pl.DataFrame()
        if key_column:
            old_columns = dfs[list(dfs.keys())[0]].collect_schema().names()
            if key_column in old_columns and not override_existing_key:
                Logger.alert1(f'key_column {key_column} already exists in dataframe columns [{old_columns}] , if you want to override it, set override_existing_key to True')
            else:
                dfs = {key:df.with_columns(pl.lit(key).alias(key_column)) for key,df in dfs.items()}
        df_list : list[Any] = list(dfs.values())
        df = pl.concat(df_list , how = 'diagonal_relaxed')
        if isinstance(df , pl.LazyFrame):
            df = df.collect()
    return dfHandler.wrapped_mapper(mapper)(df)

# def load_df_one_pl(path : PATH_TYPE | io.BytesIO , *, missing_ok = True , mapper : PL_MAPPER_TYPE = None):
#     """load dataframe from path"""
#     return FileIOHandler.load_polars(path , missing_ok = missing_ok , mapper = mapper)

def load_dfs_pl(
    paths : PATHS_TYPE , * ,  
    accelerator : Literal['thread'] | None = 'thread' , 
    mapper : PL_MAPPER_TYPE = None
) -> dict[int | Any, pl.DataFrame]:
    """
    load dataframe from multiple paths
    Parameters
    ----------
    paths : dict[int, Path]
        paths to load , key is date
    key_column : str | None
        key column name , if None, use date column
    accelerator : Literal['thread'] | None
        accelerating mode
    mapper : Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pl.DataFrame] | None
        mapper function to execute on each dataframe
    """
    return dfIOHandler.load_polars_multiple(paths , accelerator = accelerator , mapper = dfHandler.wrapped_mapper(mapper))

def save(df : pd.DataFrame | None , db_src : str , db_key : str , date = None , *, 
         overwrite = True , indent = 1 , vb_level : Any = 1 , reason : str = ''):
    '''
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
    '''
    df = dfHandler.reset_index_pandas(df , reset = True)
    db_path = DBPath(db_src , db_key)
    mark = save_df(df , db_path.path_exact(date) , overwrite = overwrite , prefix = f'{db_src.title()} {reason}' if reason else db_key , 
                   indent = indent , vb_level = vb_level)
    return mark

def load(db_src : str , db_key : str , date : int | None = None , *, 
         key_column = None , use_alt = False , closest = False , 
         missing_ok = True , indent = 1 , vb_level : Any = 1 , **kwargs) -> pd.DataFrame: 
    '''
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
    '''
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
          key_column = 'date' , override_existing_key = False , use_alt = False , 
          accelerator : Literal['thread' , 'process' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
          fill_datavendor = False , indent = 1 , vb_level : Any = 1 , **kwargs):
    """load multiple dates from database"""
    assert DBPath.ByDate(db_src) , f'{db_src}.{db_key} is a name database, use load instead'

    db_path = DBPath(db_src , db_key)
    if dates is None:
        assert start is not None or end is not None , f'start or end must be provided if dates is not provided'
        dates = db_path.dates(start , end , use_alt = use_alt)
    paths = {int(date):db_path.path(date , use_alt = use_alt) for date in dates}
    if not paths:
        return pd.DataFrame()
    df = load_df(paths , key_column = key_column , override_existing_key = override_existing_key , accelerator = accelerator)
    df = dfHandler.load_process_pandas(df , syntax = db_path.syntax(dates) , indent = indent , vb_level = vb_level , **kwargs)
    if fill_datavendor:
        from src.data.loader import DATAVENDOR
        DATAVENDOR.db_loads_callback(df , db_src , db_key)
    return df

def loads_pl(db_src : str , db_key : str , dates : np.ndarray | list[int] | None = None , start : int | None = None , end : int | None = None , *,
             key_column : str | None = 'date' , override_existing_key = False , use_alt = False , 
             accelerator : Literal['thread' , 'lazy'] | None = 'thread' , 
             fill_datavendor = False , indent = 1 , vb_level : Any = 1 , **kwargs):
    """load multiple dates from database but use polars to load"""
    assert DBPath.ByDate(db_src) , f'{db_src}.{db_key} is a name database, use load_pl instead'
    db_path = DBPath(db_src , db_key)
    if dates is None:
        assert start is not None or end is not None , f'start or end must be provided if dates is not provided'
        dates = db_path.dates(start , end , use_alt = use_alt)
    paths = {int(date):db_path.path(date , use_alt = use_alt) for date in dates}
    df = load_df_pl(paths , key_column = key_column , override_existing_key = override_existing_key , accelerator = accelerator)
    df = dfHandler.load_process_polars(df , syntax = db_path.syntax(dates) , indent = indent , vb_level = vb_level , **kwargs)
    if fill_datavendor:
        from src.data.loader import DATAVENDOR
        DATAVENDOR.db_loads_callback(df , db_src , db_key)
    return df

def load_df_max_date(path : PATH_TYPE , key_column : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 19000101
    else:
        return int(max(df[key_column]))

def load_df_min_date(path : PATH_TYPE , key_column : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 99991231
    else:
        return int(min(df[key_column]))