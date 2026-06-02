"""Async save utilities"""
from __future__ import annotations
import concurrent.futures
import torch
import numpy as np
import pandas as pd
import polars as pl
import threading
from datetime import datetime
from copy import deepcopy
from matplotlib.figure import Figure
from typing import Any , Callable , Mapping , TypeVar , cast
from pathlib import Path

from src.proj.core import strPath
from src.proj.db import save_df
from src.proj.log import Logger
from concurrent.futures import Future , CancelledError
from .export import dfs_to_excel , figs_to_pdf

__all__ = ['AsyncSaver']

T = TypeVar('T')

def _prepare_data(data : T , *, copy_for_safety : bool = True) -> T:
    """
    prepare torch data for async save , move to cpu and clone for safety
    for dict and list, recursively call _prepare_torch_data
    """
    if not copy_for_safety:
        return data
    if isinstance(data, torch.Tensor):
        return cast(T, data.detach().cpu().clone())
    elif isinstance(data, dict):
        return cast(T, {k: _prepare_data(v) for k, v in data.items()})
    elif isinstance(data, list):
        return cast(T, [_prepare_data(v) for v in data])
    else:
        if not isinstance(data, (pd.DataFrame, pl.DataFrame, Figure, np.ndarray)):
            Logger.warning(f'data is of type {type(data)}, will be copied for safety')
        return deepcopy(data)

def _wait_futures(futures : list[Future] , raise_first_error : bool = False) -> list[BaseException]:
    """
    wait for all futures to complete
    return a list of exceptions if any future raises an exception
    """
    errors : list[BaseException] = []

    for future in futures:
        try:
            future.result()
        except CancelledError:
            # Cancelled futures are expected in some call sites.
            continue
        except BaseException as e:
            errors.append(e)
            if raise_first_error:
                raise
    return errors

def _torch_save_with_footnote(obj : Any , path : strPath , prefix : str | None = None , indent : int = 1 , vb_level : Any = 3) -> None:
    """
    save torch object to path with footnote
    """
    torch.save(obj, path)
    if prefix:
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)

def _aprefix(prefix : str | None = None) -> str:
    """
    get the async prefix
    """
    return f'(Async) {prefix}' if prefix else ''
class AsyncSaver:
    """
    AsyncSaver is a class that provides a way to save data asynchronously using a global thread pool.

    use AsyncSaver.torch, AsyncSaver.df, AsyncSaver.dfs, AsyncSaver.figs to save data asynchronously.

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
    _executor_lock = threading.Lock()
    _futures_lock = threading.Lock()
    _futures : dict[Future , str] = {}

    @classmethod
    def executor(cls) -> concurrent.futures.ThreadPoolExecutor:
        """
        ensure the executor is created
        """
        with cls._executor_lock:
            if not hasattr(cls , '_executor'):
                cls._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        return cls._executor

    @classmethod
    def _register_future(cls , future : Future , future_group : str | None = None) -> None:
        """
        register the future with the future group
        """
        cls.clean_up_futures()
        future_group = future_group or 'unspecified'
        with cls._futures_lock:
            cls._futures[future] = future_group

        def _cleanup(done_future : Future) -> None:
            with cls._futures_lock:
                cls._futures.pop(done_future , None)
        future.add_done_callback(_cleanup)

    @classmethod
    def submit(cls , func : Callable , *args , future_group : str | None = None , **kwargs) -> Future:
        """
        submit the function to the executor
        """
        future = cls.executor().submit(func , *args , **kwargs)
        cls._register_future(future , future_group)
        return future

    @classmethod
    def clean_up_futures(cls) -> None:
        """
        remove all futures that are done or cancelled
        """
        with cls._futures_lock:
            for future in cls._futures:
                if future.done() or future.cancelled():
                    cls._futures.pop(future)

    @classmethod
    def pending_futures(cls , future_group : str | None = None) -> list[Future]:
        """
        get the pending futures for the future group
        """
        cls.clean_up_futures()
        with cls._futures_lock:
            if future_group:
                return [future for future in cls._futures if cls._futures[future] == future_group]
            else:
                return [future for future in cls._futures]

    @classmethod
    def wait_all(cls , future_group : str | None = None , raise_first_error : bool = False , caller_name : str | None = None) -> list[BaseException]:
        """
        wait for all futures in the future group to complete
        """
        futures = cls.pending_futures(future_group)
        if futures:
            time_start = datetime.now()
            exceptions = _wait_futures(futures , raise_first_error = raise_first_error)
            time_cost = datetime.now() - time_start
            if time_cost.total_seconds() > 1:
                group_str = f" in group {future_group}" if future_group else ""
                caller_str = f"{caller_name} >> " if caller_name else ""
                msg = f'{caller_str}Waited {time_cost.total_seconds():.1f} seconds for {len(futures)} AsyncSaver{group_str} to complete'
                Logger.note(msg , vb_level = 'max')
            return exceptions
        else:
            return []

    @classmethod
    def cancel_all(cls , future_group : str | None = None) -> None:
        """
        cancel all futures in the future group
        """
        futures = cls.pending_futures(future_group)
        for future in futures:
            future.cancel()

    @classmethod
    def shutdown(cls , wait : bool = True , cancel_futures : bool = False) -> None:
        """
        shutdown the executor
        """
        with cls._executor_lock:
            if hasattr(cls , '_executor'):
                cls._executor.shutdown(wait=wait , cancel_futures=cancel_futures)

        with cls._futures_lock:
            cls._futures.clear()

    @classmethod
    def torch(
        cls , 
        obj : Any , path : strPath , copy_for_safety : bool = True , * , 
        future_group : str | None = None , prefix : str | None = None , **kwargs
    ) -> Future:
        """
        async save torch object to path
        Args:
            obj: torch object to save
            path: save path
            copy_for_safety: if True, copy the object for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            **kwargs: kwargs for torch.save
        
        Returns:
            future: concurrent.futures.Future object
        """
        obj = _prepare_data(obj, copy_for_safety = copy_for_safety)
        kwargs = kwargs | {'prefix': _aprefix(prefix) , 'future_group': future_group}
        future = cls.submit(_torch_save_with_footnote, obj, path, **kwargs)
        return future

    @classmethod
    def df(
        cls , 
        df : pd.DataFrame | pl.DataFrame , path : strPath , copy_for_safety : bool = True , * ,
        future_group : str | None = None , prefix : str | None = None , **kwargs
    ) -> Future:
        """
        async save dataframe to path
        Args:
            df: dataframe to save
            path: save path
            copy_for_safety: if True, copy the dataframe for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            **kwargs: kwargs for save_df
        
        Returns:
            future: concurrent.futures.Future object
        """
        df = _prepare_data(df, copy_for_safety = copy_for_safety)
        kwargs = kwargs | {'footnote': True , 'prefix': _aprefix(prefix) , 'future_group': future_group}
        future = cls.submit(save_df, df, path, **kwargs)
        return future

    @classmethod
    def dfs(
        cls , 
        dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , copy_for_safety : bool = False , * ,
        future_group : str | None = None , prefix : str | None = None , **kwargs
    ) -> Future:
        """
        async save multiple dataframes to path (excel format)
        
        Args:
            dfs: dictionary of dataframes to save
            path: save path
            copy_for_safety: if True, copy the dataframes for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            **kwargs: kwargs for dfs_to_excel
        Returns:
            future: concurrent.futures.Future object
        """
        dfs = _prepare_data(dfs, copy_for_safety = copy_for_safety)
        kwargs = kwargs | {'prefix': _aprefix(prefix) , 'future_group': future_group}
        future = cls.submit(dfs_to_excel, dfs, path, **kwargs)
        return future

    @classmethod
    def figs(
        cls , 
        figs , path : strPath , copy_for_safety : bool = False , * ,
        future_group : str | None = None , prefix : str | None = None , **kwargs
    ) -> Future:
        """
        async save multiple figures to path (pdf format)
        Args:
            figs: dictionary of figures to save
            path: save path
            copy_for_safety: if True, copy the figures for safety
            future_group: group name for the future , used for waiting for all futures in the group to complete or cancel
            **kwargs: kwargs for figs_to_pdf
        
        Returns:
            future: concurrent.futures.Future object
        """
        figs = _prepare_data(figs, copy_for_safety = copy_for_safety)
        kwargs = kwargs | {'prefix': _aprefix(prefix) , 'future_group': future_group}
        future = cls.submit(figs_to_pdf, figs, path, **kwargs)
        return future

    @classmethod
    def pack(
        cls , 
        source_path : strPath , target_path : strPath , overwrite = False , * ,
        future_group : str | None = None , prefix : str | None = None , **kwargs
    ) -> Future:
        """
        pack the source path to the target path
        """
        source_path , target_path = Path(source_path) , Path(target_path)
        assert source_path.exists() and source_path.is_dir() and any(source_path.iterdir()) , f'{source_path} does not exist or is empty'
        assert target_path.suffix in ['.tar' , '.tar.gz' , '.tar.bz2' , '.tar.xz' , '.tar.zst'] , f'{target_path} is not a tar file'
        if overwrite and target_path.exists():
            target_path.unlink()
        elif target_path.exists():
            raise FileExistsError(f'{target_path} already exists')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        import tarfile
        def packing(prefix : str | None = None , indent : int = 1 , vb_level : Any = 3 , **kwargs):
            with tarfile.open(target_path, 'w:gz') as tar:
                for path in source_path.iterdir():
                    tar.add(path, arcname=path.relative_to(source_path))
            if prefix:
                Logger.footnote(f'{prefix} packed to {target_path}' , indent = indent , vb_level = vb_level)
        kwargs = kwargs | {'prefix': _aprefix(prefix) , 'future_group': future_group}
        future = cls.submit(packing, **kwargs)
        return future