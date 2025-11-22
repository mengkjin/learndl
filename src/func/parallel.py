import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any , Callable , Literal

cpu_count = os.cpu_count()
MAX_WORKERS : int = min(40 , cpu_count) if cpu_count is not None else 1

def get_method(method , max_workers : int = MAX_WORKERS) -> int:
    '''
    0 : forloop
    1 : thread
    2 : process
    '''
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

TYPE_FUNC_CALL = Callable | tuple[Callable , Iterable | None] | tuple[Callable , list | tuple | None , dict | None]

def unwrap_func_call(func_call : TYPE_FUNC_CALL) -> tuple[Callable , list | tuple , dict]:
    if isinstance(func_call , Callable):
        func , args , kwargs = func_call , [] , {}
    elif isinstance(func_call , tuple):
        if len(func_call) == 2:
            if func_call[1] is None:
                func , args , kwargs = func_call[0] , [] , {}
            elif isinstance(func_call[1] , dict):
                func , args , kwargs = func_call[0] , [] , func_call[1] or {}
            elif isinstance(func_call[1] , (list,tuple)):
                func , args , kwargs = func_call[0] , func_call[1] or [] , {}
            else:
                raise ValueError(f'func_call[1] should be None , dict , list or tuple , but got {type(func_call[1])}')
        elif len(func_call) == 3:
            func , args , kwargs = func_call[0] , func_call[1] or [] , func_call[2] or {}
        else:
            raise ValueError(f'func_call should be a tuple of length 2 or 3 , but got {len(func_call)}')
    else:
        raise ValueError(f'func_call should be a Callable or a tuple , but got {type(func_call)}')
    return func , args , kwargs

def try_func_call(
    result_dict : dict , key , 
    func_call : TYPE_FUNC_CALL , 
    catch_errors : tuple[type[Exception],...] =  () , 
) -> Any:
    try:
        func , args , kwargs = unwrap_func_call(func_call)
        result_dict[key] = func(*args , **kwargs)
    except catch_errors as e:
        print(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}')
    except Exception as e:
        print(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}')
        raise e

def parallel(func : Callable , args : Iterable , kwargs : dict[Any , Any] | None = None , keys : Iterable | None = None , 
             method : str | int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
             max_workers = MAX_WORKERS , ignore_error = False):
    method = get_method(method , max_workers)
    result : dict[Any , Any] = {}
    iterance = enumerate(args) if keys is None else zip(keys , args)
    catch_errors = (Exception , ) if ignore_error else ()
    kwargs = kwargs or {}

    def try_func(result : dict , key , func : Callable , *args : Any , **kwargs : Any):
        try:
            result[key] = func(*args , **kwargs)
        except catch_errors as e:
            print(f'{key} : {args} generates an exception: {e}')
        except Exception as e:
            print(f'{key} : {args} generates an exception: {e}')
            raise e
        
    if method == 0:
        for key , arg in iterance:
            try_func(result , key , func , arg , **kwargs)
    else:
        PoolExecutor = ProcessPoolExecutor if method == 2 else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(func, arg , **kwargs):(key , arg) for key , arg in iterance}
            for future in as_completed(futures):
                try_func(result , futures[future][0] , future.result)
    return result

def parallels(
    func_calls : Iterable[TYPE_FUNC_CALL] , 
    keys : Iterable | None = None , 
    method : str | int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
    max_workers = MAX_WORKERS , ignore_error = False
):
    method = get_method(method , max_workers)
    result : dict[Any , Any] = {}
    iterance = enumerate(func_calls) if keys is None else zip(keys , func_calls)
    catch_errors = (Exception , ) if ignore_error else ()
        
    if method == 0:
        for key , func_call in iterance:
            try_func_call(result , key , func_call , catch_errors = catch_errors)
    else:
        PoolExecutor = ProcessPoolExecutor if method == 2 else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(try_func_call, result , key , func_call , catch_errors = catch_errors):key for key , func_call in iterance}
            for future in as_completed(futures):
                future.result()
    return result