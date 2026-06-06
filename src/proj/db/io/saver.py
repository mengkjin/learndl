"""Async save utilities"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Any , Literal , Mapping , TYPE_CHECKING

from src.proj.core import strPath , strPaths
from .dataframe import save_df , save_dfs_router
from .figure import figs_to_pdf
from .torch import torch_save
from .tarfile import packing

if TYPE_CHECKING:
    import torch
    import numpy as np
    import polars as pl
    from concurrent.futures import Future

__all__ = ['Save']

class Save:
    """
    Save is a class that provides a way to save data using a global thread pool.

    use Save.torch, Save.df, Save.dfs, Save.figs to save data.

    Args:
        object: torch object, dataframe, dictionary of dataframes, or dictionary of figures
        path: save path
        copy_for_safety: if True, copy the figures for safety
        future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
        **kwargs: kwargs for figs_to_pdf
    
    Returns:
        future: concurrent.futures.Future object
            call future.result() to wait for the save to complete
            call future.done() to check if the save is complete
            call future.cancel() to cancel the save
    """
    @classmethod
    def torch(
        cls , 
        obj : Any , path : strPath , async_save : bool = False , * , 
        prefix : str | None = None , future_group : str | None = None , copy_for_safety : bool = True ,  **kwargs
    ) -> Future | bool:
        """
        async save torch object to path
        Args:
            obj: torch object to save
            path: save path
            async_save: if True, save the object asynchronously
            prefix: prefix for the saver's footnote
            **kwargs: kwargs for torch.save

        AsyncSaver parameters:
            copy_for_safety: if True, copy the object for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
        
        Returns:
            bool: if saved successfully or
            future: concurrent.futures.Future object
        """
        if async_save:
            return cls.async_save('torch', obj, path, prefix = prefix, future_group = future_group, copy_for_safety = copy_for_safety, **kwargs)
        else:
            return torch_save(obj, path, prefix = prefix, **kwargs)

    @classmethod
    def df(
        cls , 
        df : pd.DataFrame | pl.DataFrame , path : strPath , async_save : bool = False , * ,
        prefix : str | None = None , future_group : str | None = None , copy_for_safety : bool = True , **kwargs
    ) -> Future | bool:
        """
        async save dataframe to path
        Args:
            df: dataframe to save
            path: save path
            async_save: if True, save the dataframe asynchronously
            prefix: prefix for the saver's footnote
            **kwargs: kwargs for save_df

        AsyncSaver parameters:
            copy_for_safety: if True, copy the dataframe for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
        
        Returns:
            bool: if saved successfully or
            future: concurrent.futures.Future object
        """
        if async_save:
            return cls.async_save('df', df, path, prefix = prefix, future_group = future_group, copy_for_safety = copy_for_safety, **kwargs)
        else:
            return save_df(df, path, prefix = prefix, footnote = True, **kwargs)

    @classmethod
    def dfs(
        cls , 
        dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , async_save : bool = False , * ,
        prefix : str | None = None , future_group : str | None = None , copy_for_safety : bool = False , 
        meta : dict[str, Any] | None = None , **kwargs
    ) -> Future | bool:
        """
        async save multiple dataframes to path (excel if suffix is .xlsx, tar if path in [.tar, .tar.gz, .tar.bz2, .tar.xz, .tar.zst])
        
        Args:
            dfs: dictionary of dataframes to save
            path: save path , should be a .xlsx file or a tar file
            async_save: if True, save the dataframes asynchronously
            prefix: prefix for the saver's footnote
            meta: metadata for the dataframes
            **kwargs: kwargs for dfs_to_excel

        AsyncSaver parameters:
            copy_for_safety: if True, copy the dataframes for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            
        Returns:
            bool: if saved successfully or
            future: concurrent.futures.Future object
        """
        if async_save:
            return cls.async_save('dfs', dfs, path, prefix = prefix, future_group = future_group, copy_for_safety = copy_for_safety, meta = meta, **kwargs)
        else:
            return save_dfs_router(dfs, path, prefix = prefix, meta = meta, **kwargs)

    @classmethod
    def figs(
        cls , 
        figs , path : strPath , async_save : bool = False , * ,
        prefix : str | None = None , future_group : str | None = None , copy_for_safety : bool = False , **kwargs
    ) -> Future | bool:
        """
        async save multiple figures to path (pdf format)
        Args:
            figs: dictionary of figures to save
            path: save path
            async_save: if True, save the figures asynchronously
            prefix: prefix for the saver's footnote
            **kwargs: kwargs for figs_to_pdf

        AsyncSaver parameters:
            copy_for_safety: if True, copy the figures for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            
        Returns:
            bool: if saved successfully or
            future: concurrent.futures.Future object
        """
        if async_save:
            return cls.async_save('figs', figs, path, prefix = prefix, future_group = future_group, copy_for_safety = copy_for_safety, **kwargs)
        else:
            return figs_to_pdf(figs, path, prefix = prefix, **kwargs)

    @classmethod
    def pack(
        cls , 
        source_path : strPath | strPaths , target_path : strPath , async_save : bool = False , * ,
        overwrite = False , prefix : str | None = None , future_group : str | None = None , **kwargs
    ) -> Future | bool:
        """
        pack the source path to the target path

        Args:
            source_path: source path
            target_path: target path
            async_save: if True, pack the source path asynchronously
            overwrite: if True, overwrite the target path
            prefix: prefix for the saver's footnote
            **kwargs: kwargs for pack_files_to_tar

        AsyncSaver parameters:
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            
        Returns:
            bool: if packed successfully or
            future: concurrent.futures.Future object
        """
        if async_save:
            return cls.async_save('pack', source_path, target_path, overwrite = overwrite, prefix = prefix, future_group = future_group, **kwargs)
        else:
            return packing(source_path, target_path, prefix = prefix, overwrite = overwrite, **kwargs)

    @classmethod
    def mmap(cls , array : np.ndarray | torch.Tensor , path : strPath):
        """save mmap to path
        Parameters
        ----------
        array : np.ndarray | torch.Tensor
            array to save
        path : strPath
            path to save mmap
        """
        from src.proj.db.io.mmap import ArrayMemoryMap
        return ArrayMemoryMap.save(array, path)

    @classmethod
    def async_save(cls , save_type : Literal['df' , 'dfs' , 'figs' , 'torch' , 'pack'] , data : Any , path : strPath , * ,
        prefix : str | None = None , future_group : str | None = None , **kwargs) -> Future:
        """
        async saving api for different types of data
        """
        from src.proj.db.io.async_saver import AsyncSaver
        match save_type:
            case 'df':
                assert data.__class__.__qualname__ in ['DataFrame'], f'data must be a pandas or polars dataframe, got {type(data)}'
            case 'dfs':
                assert isinstance(data, dict), f'data must be a dictionary of dataframes, got {type(data)}'
            case 'figs':
                assert isinstance(data, dict), f'data must be a dictionary of figures, got {type(data)}'
            case 'torch':
                ...
            case 'pack':
                assert isinstance(data, (str , Path)), f'data must be a string or pathlib.Path, got {type(data)}'
            case _:
                raise ValueError(f'Invalid type: {type}')
            
        return getattr(AsyncSaver, save_type)(data, path, prefix = prefix, future_group = future_group, **kwargs)
        
    @classmethod
    def async_clean_up_futures(cls) -> None:
        """
        remove all futures that are done or cancelled
        """
        from src.proj.db.io.async_saver import AsyncSaver
        AsyncSaver.clean_up_futures()

    @classmethod
    def async_pending_futures(cls , future_group : str | None = None) -> list[Future]:
        """
        get the pending futures for the future group
        """
        from src.proj.db.io.async_saver import AsyncSaver
        return AsyncSaver.pending_futures(future_group)

    @classmethod
    def async_wait_all(cls , future_group : str | None = None , raise_first_error : bool = False , caller_name : str | None = None) -> list[BaseException]:
        """
        wait for all futures in the future group to complete
        """
        from src.proj.db.io.async_saver import AsyncSaver
        return AsyncSaver.wait_all(future_group, raise_first_error, caller_name)

    @classmethod
    def async_cancel_all(cls , future_group : str | None = None) -> None:
        """
        cancel all futures in the future group
        """
        from src.proj.db.io.async_saver import AsyncSaver
        return AsyncSaver.cancel_all(future_group)

    @classmethod
    def async_shutdown(cls , wait : bool = True , cancel_futures : bool = False) -> None:
        """
        shutdown the executor
        """
        from src.proj.db.io.async_saver import AsyncSaver
        return AsyncSaver.shutdown(wait=wait , cancel_futures=cancel_futures)