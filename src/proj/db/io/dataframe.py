"""Load, save, and path helpers for versioned tables under ``PATH.data`` (feather/parquet, tar, parallel IO)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import io
import os
from uuid import uuid4

from pathlib import Path
from typing import Any , Literal , TYPE_CHECKING
from collections.abc import Mapping, Iterable, Callable

from src.proj.env import MACHINE
from src.proj.log import Logger
from src.proj.core import strPath , strPaths , lit
from src.proj.core.literals import PandasAccelerator , PolarsAccelerator

from src.proj.db.basic import DF_SUFFIX , path_date , dfHandler

if TYPE_CHECKING:
    import polars as pl
    PL_MAPPER_TYPE = Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pl.DataFrame] | None
    PD_MAPPER_TYPE = Iterable[Callable[[pd.DataFrame], pd.DataFrame]] | Callable[[pd.DataFrame], pd.DataFrame] | None
    
__all__ = [
    'save_df' , 'append_df' , 
    'load_df' , 'load_dfs' , 'load_df_pl' , 'load_dfs_pl' , 
    'load_df_max_date' , 'load_df_min_date' , 'dfs_to_excel'
]

class dfIOHandler:
    """File IO operations handler"""
    @classmethod
    def load_pandas(
        cls , path : strPath | io.BytesIO , * , 
        missing_ok = True , 
        mapper : PD_MAPPER_TYPE = None
    ) -> pd.DataFrame:
        """load dataframe from path"""
        if isinstance(path , strPath) and not Path(path).exists() and missing_ok: 
            return pd.DataFrame()
        try:
            if DF_SUFFIX == 'feather':
                df = pd.read_feather(path)
            else:
                df = pd.read_parquet(path , engine='fastparquet')
        except Exception as e:
            Logger.error(f'Error loading {path}: {e}')
            raise
        df = dfHandler.apply_mapper(df , mapper)
        return df

    @classmethod
    def load_polars(cls , path : strPath | io.BytesIO , * , missing_ok = True , mapper : PL_MAPPER_TYPE = None) -> pl.DataFrame:
        """load dataframe from path"""
        import polars as pl
        if isinstance(path , strPath) and not Path(path).exists() and missing_ok: 
            return pl.DataFrame()
        if DF_SUFFIX == 'feather':
            df = pl.read_ipc(path , memory_map = False)
        else:
            df = pl.read_parquet(path , memory_map = False)
        df = dfHandler.apply_mapper(df , mapper)
        return df
    
    @classmethod
    def save_df(cls , df : pd.DataFrame | pl.DataFrame , path : strPath | io.BytesIO):
        """save dataframe to path"""
        import polars as pl
        if isinstance(df , pd.DataFrame) and None in df.index.names:
            df = df.reset_index(None , drop = True)
        # Atomic write for filesystem paths: write to a temp file in the same directory,
        # then replace the target path. Prevents partially-written files on crashes.
        tmp_path: Path | None = None
        target_path: Path | None = None
        if isinstance(path, (str, Path)):
            target_path = Path(path)
            tmp_path = target_path.with_name(
                f"{target_path.name}.tmp.{os.getpid()}.{uuid4().hex[:8]}"
            )
        try:
            write_path = tmp_path if tmp_path is not None else path
            if isinstance(df , pd.DataFrame) and DF_SUFFIX == 'feather':
                df.to_feather(write_path)
            elif isinstance(df , pd.DataFrame) and DF_SUFFIX == 'parquet':
                df.to_parquet(write_path , engine='fastparquet')
            elif isinstance(df , pl.DataFrame) and DF_SUFFIX == 'feather':
                df.write_ipc(write_path)
            elif isinstance(df , pl.DataFrame) and DF_SUFFIX == 'parquet':
                df.write_parquet(write_path)
            else:
                raise ValueError(f'Unsupported dataframe type {type(df)} with suffix {DF_SUFFIX}')
            if tmp_path is not None and target_path is not None:
                os.replace(tmp_path, target_path)
        except Exception as e:
            Logger.error(f'Error saving {path}: {e}')
            Logger.display(df , title = 'Error saving DataFrame')
            raise
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    ...

    @classmethod
    def to_path_dict(cls , paths : strPaths) -> dict[int | Any, Path]:
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
        cls , paths : strPaths , * ,
        accelerator : PandasAccelerator | None = 'thread' , 
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

        def loader(p : strPath) -> pd.DataFrame:
            return cls.load_pandas(p , mapper = mapper)
        if accelerator is None:
            dfs = {d:loader(p) for d,p in paths.items() if not loader(p).empty}
        elif accelerator == 'dask':
            from dask.delayed import delayed
            from dask.base import compute
            ddfs = [delayed(loader)(p) for d,p in paths.items()]
            dfs = {d:df for d,df in zip(paths.keys() , compute(ddfs)[0])}
        elif accelerator == 'thread':
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(MACHINE.max_workers , max(len(paths) // 5 , 1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(loader , p):d for d,p in paths.items()}
                dfs = {futures[future]:future.result() for future in as_completed(futures)}
        else:
            raise ValueError(f'Unsupported accelerator: {accelerator}')
        return dfs

    @classmethod
    def load_polars_multiple(
        cls , paths : strPaths , * , 
        accelerator : PolarsAccelerator | None = 'thread' , 
        mapper : PL_MAPPER_TYPE = None ,
    ) -> dict[int | Any, pl.DataFrame]:
        """
        load dataframe from multiple paths in accelerating mode
        """
        def loader(p : strPath) -> pl.DataFrame:
            return cls.load_polars(p , mapper = mapper)

        paths = {d:p for d,p in cls.to_path_dict(paths).items() if p.exists()}
        if not paths:
            return {}
        if accelerator is None:
            dfs = {d:loader(p) for d,p in paths.items()}
        elif accelerator == 'lazy':
            dfs = {d:pl.scan_ipc(p) for d,p in paths.items()}
            dfs = {d:df.collect() for d,df in dfs.items()}
        elif accelerator == 'thread':
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(MACHINE.max_workers , max(len(paths) // 5 , 1))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(loader , p):d for d,p in paths.items()}
                dfs = {futures[future]:future.result() for future in as_completed(futures)}
        else:
            raise ValueError(f'Unsupported accelerator: {accelerator}')
        return dfs
  

def save_df(
    df : pd.DataFrame | pl.DataFrame | None , path : strPath , *, 
    overwrite = True , prefix : str | None = None , empty_ok = False , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 1 , footnote = False):
    """save dataframe to path"""
    if df is None or (not empty_ok and len(df) == 0): 
        return False
    prefix = prefix or ''
    path = Path(path)
    if overwrite or not path.exists(): 
        status = 'Overwritten ' if path.exists() else 'File Created'
        path.parent.mkdir(parents=True , exist_ok=True)
        dfIOHandler.save_df(df , path)
        if footnote:
            Logger.footnote(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        else:
            Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)
        return True
    else:
        status = 'File Exists '
        Logger.alert1(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        return False

def append_df(
    df : pd.DataFrame | None , path : strPath , * , 
    drop_duplicate_cols : list[str] | None = None , prefix : str | None = None , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 1 , footnote = False):
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
        if footnote:
            Logger.footnote(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level)
        else:
            Logger.stdout(f'{prefix} {status}: {path}' , indent = indent , vb_level = vb_level , italic = True)

def load_df(
    path : strPath | strPaths , * , 
    missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
    accelerator : PandasAccelerator | None = 'thread' , 
    mapper : PD_MAPPER_TYPE = None
):
    """
    load dataframe from path or paths
    Parameters
    ----------
    path : strPath | strPaths
        path or paths to load , key is date
    missing_ok : bool
        if True, return empty dataframe for missing path(s)
    key_column : str | None
        key column name , if None, use date column
    accelerator : 'thread' | 'dask' | 'polars' | 'polars_thread' | None
        accelerating mode
    mapper : Callable[[pd.DataFrame], pd.DataFrame]
        mapper function to execute on each dataframe
    """
    if isinstance(path , strPath):
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
    paths : strPath | strPaths , * ,  
    accelerator : PandasAccelerator | None = 'thread' , 
    mapper : PD_MAPPER_TYPE = None , **kwargs
) -> dict[int | Any, pd.DataFrame]:
    """
    load dataframe from multiple paths , return dict of date and dataframe
    Parameters
    ----------
    paths : strPaths | strPath
        paths to load , key is date
    accelerator : 'thread' | 'dask' | 'polars' | 'polars_thread' | None
        accelerating mode
    mapper : Iterable[Callable[[pd.DataFrame], pd.DataFrame]] | Callable[[pd.DataFrame], pd.DataFrame] | None
        mapper function to execute on each dataframe
    """
    if isinstance(paths , strPath):
        paths = [paths]
    return dfIOHandler.load_pandas_multiple(paths , accelerator = accelerator , mapper = dfHandler.wrapped_mapper(mapper))

def load_df_pl(
    path : strPath | strPaths , *, 
    missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
    accelerator : PolarsAccelerator | None = 'thread' , 
    mapper : PL_MAPPER_TYPE = None
) -> pl.DataFrame:
    """
    load polars dataframe from path or paths
    Parameters
    ----------
    path : strPath | strPaths
        path or paths to load , key is date
    missing_ok : bool
        if True, return empty dataframe for missing path(s)
    accelerator : 'thread' | 'lazy' | None
        accelerating mode
    mapper : Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pd.DataFrame] | None
        mapper function to execute on each dataframe
    """
    import polars as pl
    if isinstance(path , strPath):
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

def load_dfs_pl(
    paths : strPaths , * ,  
    accelerator : PolarsAccelerator | None = 'thread' , 
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
    accelerator : 'thread' | 'lazy' | None
        accelerating mode
    mapper : Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pd.DataFrame] | None
        mapper function to execute on each dataframe
    """
    return dfIOHandler.load_polars_multiple(paths , accelerator = accelerator , mapper = dfHandler.wrapped_mapper(mapper))

def load_df_max_date(path : strPath , key_column : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 19000101
    else:
        return int(max(df[key_column]))

def load_df_min_date(path : strPath , key_column : str = 'date') -> int:
    """load dataframe from path"""
    path = Path(path)
    if not path.exists() or (df := load_df(path)).empty:
        return 99991231
    else:
        return int(min(df[key_column]))

def dfs_to_excel(
    dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , 
    mode : Literal['a','w'] = 'w' , sheet_prefix = '' , prefix : str | None = None , 
    indent : int = 1 , vb_level : lit.VerbosityLevel = 3
):
    """Write each DataFrame to a sheet; optionally log via ``Logger.footnote``.

    Returns:
        Output ``path``.
    """
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    if mode == 'a': 
        mode = 'a' if os.path.exists(path) else 'w'
    with pd.ExcelWriter(path , 'openpyxl' , mode = mode) as writer:
        for key, value in dfs.items():
            if not isinstance(value , pd.DataFrame):
                value = value.to_pandas()
            value.to_excel(writer, sheet_name = f'{sheet_prefix}{key}')
    if prefix: 
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path

def save_dfs_router(
    dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath ,  * ,
    prefix : str | None = None , meta : dict[str, Any] | None = None , **kwargs) -> bool:
    """
    route the saving of multiple dataframes to the appropriate function
    """
    path = Path(path)
    if path.suffix == '.xlsx':
        if meta:
            Logger.alert1(f'{prefix} saving to {path} with meta: {meta} , please check if you should use tar instead')
        dfs_to_excel(dfs, path, prefix = prefix, **kwargs)
    elif path.suffix in ['.tar', '.tar.gz', '.tar.bz2', '.tar.xz', '.tar.zst']:
        from src.proj.db.io.tarfile import save_dfs_to_tar
        save_dfs_to_tar(dfs, path, meta = meta, prefix = prefix, **kwargs)
    else:
        raise ValueError(f'Unsupported suffix {path.suffix} for dfs save')
    return True
