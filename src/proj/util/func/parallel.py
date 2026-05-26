"""Run many callables in-process, threaded, or multiprocessed with shared result dict."""
from __future__ import annotations

import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from typing import Any , Callable , Literal , Mapping , Iterable , TypeVar
from uuid import uuid4

from src.proj import MACHINE , Logger
from src.proj.util.catcher import MPOutputCatcher

__all__ = ['parallel' , 'FuncCall' , 'is_main_process']

MAX_WORKERS : int = min(40 , MACHINE.cpu_count)
T = TypeVar('T')
INPUT_TYPE = (
    Callable[..., T] 
    | tuple[Callable[..., T], Iterable[Any] | None] 
    | tuple[Callable[..., T], list[Any] | tuple[Any, ...] | None, dict[str, Any] | None]
)


def is_main_process() -> bool:
    """True only in the process that launched the job (not a ``ProcessPoolExecutor`` worker)."""
    return mp.current_process().name == 'MainProcess'


def get_method(method : int | bool | Literal['forloop' , 'thread' , 'process'] , max_workers : int = MAX_WORKERS) -> int:
    """get parallel method index
    0 : forloop
    1 : thread
    2 : process
    """
    if max_workers <= 1: 
        return 0
    elif isinstance(method , bool):
        return 1 if method else 0
    elif isinstance(method , int):
        assert 0 <= method < 3 , f'method should be 0 ~ 2 , but got {method}'
        return method
    elif isinstance(method , str):
        return ['forloop' , 'thread' , 'process'].index(method)
    else:
        raise ValueError(f'method should be int or str , but got {type(method)}')

def parallel(
    inputs : Mapping[Any , INPUT_TYPE[T]] | Iterable[INPUT_TYPE[T]] , 
    method : int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
    max_workers = MAX_WORKERS , ignore_error = False , timeout : float = -1 , indent : int = 0 ,
    capture_mp_output : bool = False , keep_mp_output_on_error : bool = False , **kwargs
) -> dict[Any , T]:
    """Execute ``FuncCall`` inputs; populate and return the shared results dict.

    Args:
        inputs: Mapping or sequence of ``(func)`` or ``(func, args[, kwargs])`` specs.
        method: for-loop, thread pool, or process pool (ignored for a single call).
        max_workers: Worker cap for pool executors.
        ignore_error: If True, swallow exceptions per call and log them.
        capture_mp_output: When ``method='process'``, capture worker logs to disk and merge into
            ``HtmlCatcher.PrimaryInstance`` after the pool shuts down.
        keep_mp_output_on_error: If True, keep ``logs/catcher/mp_output/{run_id}`` when merge finds no files.

    Returns:
        Dict filled by successful ``try_call`` executions.
    """
    result , func_calls = FuncCall.from_func_calls(inputs , ignore_error = ignore_error , **kwargs)
    method = get_method(method if len(func_calls) > 1 else 0 , max_workers)
    
    if method == 0:
        remaining_timeout = timeout * 3600
        start_time = datetime.now()
        for func_call in func_calls:
            func_call()
            remaining_timeout = remaining_timeout - (datetime.now() - start_time).total_seconds()
            if remaining_timeout <= 0 and timeout > 0:
                Logger.alert1(f'Timeout reached, {timeout} hours passed, stopping parallel', indent = indent)
                break
    elif method == 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [(func_call , func_call.submit(pool)) for func_call in func_calls]
            for func_call , future in futures:
                # Thread pool: child can mutate parent ``result_dict``; use Future return value.
                returned = future.result()
    elif method == 2:
        mp_run_id = uuid4().hex[:12] if capture_mp_output else None
        mp_kwargs = {'initializer': MPOutputCatcher.pool_initializer,'initargs': (mp_run_id ,)}
        try:
            with ProcessPoolExecutor(max_workers=max_workers , **mp_kwargs) as pool:
                futures = [(func_call , func_call.submit(pool)) for func_call in func_calls]
                for func_call , future in futures:
                    # Process pool: child cannot mutate parent ``result_dict``; use Future return value.
                    returned = future.result()
                    if func_call.key is not None:
                        result[func_call.key] = returned
        finally:
            MPOutputCatcher.merge_into_html(mp_run_id , keep_on_error=keep_mp_output_on_error)
    return result

class FuncCall:
    """Wraps one callable spec with optional result capture and error policy."""

    def __init__(self , func_input : INPUT_TYPE , key : Any , 
                 result_dict : dict , catch_errors : tuple[type[Exception],...] = () ,
                 **kwargs):
        """Store call specification, result bucket key, and exception types to swallow."""
        self.func_input = func_input
        self.key = key
        self.result_dict = result_dict
        self.catch_errors = catch_errors
        self.kwargs = kwargs
       
    def __repr__(self):
        return f'FuncCall(key={self.key},func_input={self.func_input})'

    def __call__(self) -> Any:
        """Run ``go`` (same as explicit ``go()``)."""
        return self.go()

    @staticmethod
    def unwrap(func_input : INPUT_TYPE[T] , **kwargs) -> tuple[Callable[..., T] , tuple[Any, ...] , dict[str, Any]]:
        """Normalize ``func_input`` into ``(callable, args, kwargs)``."""
        args = []
        if isinstance(func_input , Callable):
            func = func_input
        elif isinstance(func_input , tuple):
            func = func_input[0]
            if len(func_input) == 2:
                obj = func_input[1]
                if obj is None:
                    ...
                elif isinstance(obj , dict):
                    kwargs = kwargs | obj
                elif isinstance(obj , (list,tuple)):
                    args = list(obj)
                else:
                    raise ValueError(f'func_input[1] should be None , dict , list or tuple , but got {type(obj)}')
            elif len(func_input) == 3:
                args , kwargs = list(func_input[1] or []) , kwargs | (func_input[2] or {})
            else:
                raise ValueError(f'func_input should be a tuple of length 2 or 3 , but got {len(func_input)}')
        else:
            raise ValueError(f'func_input should be a Callable or a tuple , but got {type(func_input)}')
        return func , tuple(args) , kwargs

    @classmethod
    def try_call(cls , func_input : INPUT_TYPE[T] , key : Any | None = None , 
                 result_dict : dict | None = None , catch_errors : tuple[type[Exception],...] = () ,
                 **kwargs) -> T | None:
        """Invoke unwrapped func; store return in ``result_dict[key]`` on success."""
        func , args , kwargs = cls.unwrap(func_input , **kwargs)
        try:
            result = func(*args , **kwargs)
            if result_dict is not None and key is not None:
                result_dict[key] = result
            return result
        except catch_errors as e:
            Logger.alert1(f'{key} >> {func}({args} , {kwargs}) generate an exception: {e}')
            Logger.print_exc(e)
        except Exception as e:
            Logger.error(f'{key} >> {func}({args} , {kwargs}) generate an exception: {e}')
            Logger.print_exc(e)
            from src.proj.cal.trade_date import BC
            Logger.info('BC._cd_pd_index.is_unique:' , BC._cd_pd_index.is_unique)
            Logger.info('BC._cd_pd_index.duplicated:' , BC._cd_pd_index[BC._cd_pd_index.duplicated()])
            raise

    def go(self) -> Any:
        """Execute this instance's bound call with instance kwargs."""
        return self.try_call(self.func_input , self.key , self.result_dict , self.catch_errors , **self.kwargs)

    def submit(self , pool : ThreadPoolExecutor | ProcessPoolExecutor):
        """Submit ``go`` to an executor; returns the Future."""
        return pool.submit(self.go)

    @classmethod
    def from_func_calls(cls , inputs : Mapping[Any , INPUT_TYPE[T]] | Iterable[INPUT_TYPE[T]] , 
                        ignore_error : bool = False , **kwargs) -> tuple[dict[Any , T] , list[FuncCall]]:
        """Build result dict and ``FuncCall`` list from a mapping or iterable."""
        iterance = inputs.items() if isinstance(inputs , dict) else enumerate(inputs)
        result = {}
        catch_errors = (Exception , ) if ignore_error else ()
        func_calls : list[FuncCall] = []
        for key , input in iterance:
            func_calls.append(FuncCall(input , key , result , catch_errors = catch_errors , **kwargs))
        return result , func_calls
