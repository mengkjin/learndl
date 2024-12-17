import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any , Callable , Literal , Optional

cpu_count = os.cpu_count()
MAX_WORKERS : int = min(40 , cpu_count) if cpu_count is not None else 1


def parallel(func : Callable , args : Iterable , keys : Optional[Iterable] = None , 
             type : Literal['thread' , 'process'] | bool | None = None , max_workers : int = 10 , ignore_error = False):
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
        
    if not type or max_workers == 1:
        for key , arg in iterance:
            try_func(result , key , func , arg)
    else:
        PoolExecutor = ProcessPoolExecutor if type == 'process' else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(func, arg):(key , arg) for key , arg in iterance}
            for future in as_completed(futures):
                key , arg = futures[future]
                try_func(result , key , future.result)
    return result