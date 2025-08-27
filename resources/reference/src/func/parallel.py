import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any , Callable , Literal , Optional

cpu_count = os.cpu_count()
MAX_WORKERS : int = min(40 , cpu_count) if cpu_count is not None else 1

def get_method(method , max_workers : int = MAX_WORKERS) -> int:
    '''
    0 : forloop
    1 : thread
    2 : process
    '''
    if max_workers <= 1: return 0
    elif isinstance(method , bool):
        return 1 if method else 0
    elif isinstance(method , int):
        assert 0 <= method < 3 , f'method should be 0 ~ 2 , but got {method}'
        return method
    elif isinstance(method , str):
        return ['forloop' , 'thread' , 'process'].index(method)
    else:
        raise ValueError(f'method should be int or str , but got {type(method)}')

def parallel(func : Callable , args : Iterable , keys : Optional[Iterable] = None , 
             method : str | int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , max_workers = MAX_WORKERS , ignore_error = False):
    method = get_method(method , max_workers)
    result : dict[Any , Any] = {}
    iterance = enumerate(args) if keys is None else zip(keys , args)
    catch_errors = (Exception , ) if ignore_error else ()

    def try_func(result : dict , key , func : Callable , *args : Any):
        try:
            result[key] = func(*args)
        except catch_errors as e:
            print(f'{key} : {args} generated an exception: {e}')
        except Exception as e:
            print(f'{key} : {args} generated an exception:')
            raise e
        
    if method == 0:
        for key , arg in iterance:
            try_func(result , key , func , arg)
    else:
        PoolExecutor = ProcessPoolExecutor if method == 2 else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(func, arg):(key , arg) for key , arg in iterance}
            for future in as_completed(futures):
                key , arg = futures[future]
                try_func(result , key , future.result)
    return result