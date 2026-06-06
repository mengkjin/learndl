"""Loader for database for different file types"""

from __future__ import annotations

from pathlib import Path
from typing import Any , Iterable , Callable , Union , Literal , TYPE_CHECKING

from src.proj.core import strPath , strPaths
from src.proj.db.basic import TAR_SUFFIXES

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    PD_MAPPER_TYPE = Union[Iterable[Callable[[pd.DataFrame], pd.DataFrame]] , Callable[[pd.DataFrame], pd.DataFrame] , None]
    PL_MAPPER_TYPE = Union[Iterable[Callable[[pl.DataFrame], pl.DataFrame]] , Callable[[pl.DataFrame], pl.DataFrame] , None]

class Load:
    """Loader for database"""
    @classmethod
    def df(cls , path : strPath | strPaths , * , missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
        accelerator : Literal['thread' , 'dask' , 'polars' , 'polars_thread'] | None = 'thread' , 
        mapper : PD_MAPPER_TYPE = None) -> pd.DataFrame:
        """
        load dataframe from path or paths
        Parameters
        ----------
        path : strPath | Iterable[strPath] | dict[int, strPath]
            path or paths to load , key is date
        missing_ok : bool
            if True, return empty dataframe for missing path(s)
        key_column : str | None
            key column name , if None, use date column
        accelerator : Literal['thread' , 'dask' , 'polars' , 'polars_thread']
            accelerating mode
        mapper : Callable[[pd.DataFrame], pd.DataFrame]
            mapper function to execute on each dataframe
        """
        from src.proj.db.io.dataframe import load_df
        return load_df(
            path,
            missing_ok=missing_ok, key_column=key_column,
            override_existing_key=override_existing_key,
            accelerator=accelerator, mapper=mapper,
        )
    
    @classmethod
    def pandas(
        cls,
        path: strPath | strPaths,
        *,
        missing_ok: bool = True,
        key_column: str | None = 'date',
        override_existing_key: bool = False,
        accelerator: Literal['thread', 'dask', 'polars', 'polars_thread'] | None = 'thread',
        mapper: PD_MAPPER_TYPE = None,
    ) -> pd.DataFrame:
        """
        Alias for Load.df
        """
        return cls.df(
            path,
            missing_ok=missing_ok, key_column=key_column,
            override_existing_key=override_existing_key,
            accelerator=accelerator, mapper=mapper,
        )

    @classmethod
    def polars(cls , path : strPath | strPaths , * , missing_ok = True , key_column : str | None = 'date' , override_existing_key = False ,
        accelerator : Literal['thread' , 'lazy'] | None = 'thread' , 
        mapper : PL_MAPPER_TYPE = None) -> pl.DataFrame:
        """
        load polars dataframe from path or paths
        Parameters
        ----------
        path : strPath | Iterable[strPath] | dict[int, strPath]
            path or paths to load , key is date
        missing_ok : bool
            if True, return empty dataframe for missing path(s)
        accelerator : Literal['thread' , 'lazy'] | None
            accelerating mode
        mapper : Iterable[Callable[[pl.DataFrame], pl.DataFrame]] | Callable[[pl.DataFrame], pl.DataFrame] | None
            mapper function to execute on each dataframe
        """
        from src.proj.db.io.dataframe import load_df_pl
        return load_df_pl(
            path , 
            missing_ok = missing_ok , key_column = key_column , 
            override_existing_key = override_existing_key , 
            accelerator = accelerator , mapper = mapper)

    @classmethod
    def torch(cls , path : strPath , weights_only : bool = False , **kwargs):
        """``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6."""
        from src.proj.db.io.torch import torch_load
        return torch_load(path , weights_only = weights_only , **kwargs)

    @classmethod
    def dfs(cls , path : strPath | strPaths , * , missing_ok = True , mapper : PD_MAPPER_TYPE = None , **kwargs) -> dict[Any , pd.DataFrame]:
        """load dataframes from tarfile , single / multiple dataframes
        Parameters
        ----------
        path : strPath | strPaths
            path to tar file
        """
        if isinstance(path , strPath) and Path(path).suffix in TAR_SUFFIXES:
            from src.proj.db.io.tarfile import load_dfs_from_tar
            return load_dfs_from_tar(path , missing_ok = missing_ok , mapper = mapper , **kwargs)
        else:
            from src.proj.db.io.dataframe import load_dfs
            return load_dfs(path , missing_ok = missing_ok , mapper = mapper , **kwargs)

    @classmethod
    def tar_meta(cls , path : strPath , **kwargs) -> dict[str, Any]:
        """load meta from tarfile
        Parameters
        ----------
        path : strPath
            path to tar file
        """
        from src.proj.db.io.tarfile import load_tar_meta
        return load_tar_meta(path)

    @classmethod
    def unpack(cls , path : strPath , target : strPath , * , overwrite = False , indent = 1 , vb_level : Any = 1):
        """unpack tar file to target directory
        Parameters
        ----------
        path : strPath
            path to tar file
        target : strPath
            target directory
        overwrite : bool
            if True, overwrite the target directory
        """
        from src.proj.db.io.tarfile import unpack_files_from_tar
        return unpack_files_from_tar(path , target , overwrite = overwrite , indent = indent , vb_level = vb_level)

    @classmethod
    def mmap(cls , path : strPath , **kwargs):
        """load mmap from path
        Parameters
        ----------
        path : strPath
            path to mmap file
        """
        from src.proj.db.io.mmap import ArrayMemoryMap
        return ArrayMemoryMap.load(path)