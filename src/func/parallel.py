import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any , Callable , Literal , Mapping

from src.proj import Logger

cpu_count = os.cpu_count()
MAX_WORKERS : int = min(40 , cpu_count) if cpu_count is not None else 1

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

INPUT_TYPE = Callable | tuple[Callable , Iterable | None] | tuple[Callable , list | tuple | None , dict | None]

def parallel(
    inputs : Mapping[Any , INPUT_TYPE] | Iterable[INPUT_TYPE] , 
    method : int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
    max_workers = MAX_WORKERS , ignore_error = False , **kwargs
):
    result , func_calls = FuncCall.from_func_calls(inputs , ignore_error = ignore_error , **kwargs)
    method = get_method(method if len(func_calls) > 1 else 0 , max_workers)

    if method == 0:
        [func_call() for func_call in func_calls]
    else:
        PoolExecutor = ProcessPoolExecutor if method == 2 else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = [func_call.submit(pool) for func_call in func_calls]
            for future in as_completed(futures):
                future.result()
    return result

class FuncCall:
    def __init__(self , func_input : INPUT_TYPE , key : Any , 
                 result_dict : dict , catch_errors : tuple[type[Exception],...] = () ,
                 **kwargs):
        self.func_input = func_input
        self.key = key
        self.result_dict = result_dict
        self.catch_errors = catch_errors
        self.kwargs = kwargs
       
    def __repr__(self):
        return f'FuncCall(key={self.key},func_input={self.func_input})'

    def __call__(self) -> Any:
        return self.do()

    @staticmethod
    def unwrap(func_input : INPUT_TYPE , **kwargs) -> tuple[Callable , list , dict]:
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
        return func , args , kwargs

    @classmethod
    def try_call(cls , func_input : INPUT_TYPE , key : Any | None = None , 
                 result_dict : dict | None = None , catch_errors : tuple[type[Exception],...] = () ,
                 **kwargs) -> Any:
        func , args , kwargs = cls.unwrap(func_input , **kwargs)
        try:
            result = func(*args , **kwargs)
            if result_dict is not None and key is not None:
                result_dict[key] = result
            return result
        except catch_errors as e:
            Logger.alert(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}' , level = 1)
        except Exception as e:
            Logger.error(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}')
            raise e

    def do(self) -> Any:
        self.try_call(self.func_input , self.key , self.result_dict , self.catch_errors , **self.kwargs)

    def submit(self , pool : ThreadPoolExecutor | ProcessPoolExecutor):
        return pool.submit(self.do)

    @classmethod
    def from_func_calls(cls , inputs : Mapping[Any , INPUT_TYPE] | Iterable[INPUT_TYPE] , ignore_error : bool = False , **kwargs) -> tuple[dict[Any , Any] , list['FuncCall']]:
        iterance = inputs.items() if isinstance(inputs , dict) else enumerate(inputs)
        result : dict[Any , Any] = {}
        catch_errors = (Exception , ) if ignore_error else ()
        func_calls : list[FuncCall] = []
        for key , input in iterance:
            func_calls.append(FuncCall(input , key , result , catch_errors = catch_errors , **kwargs))
        return result , func_calls

    @classmethod
    def parallel(cls , inputs : Mapping[Any , INPUT_TYPE] , 
                 method : int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
                 max_workers = MAX_WORKERS , ignore_error : bool = False) -> dict[Any , Any]:
        return parallel(inputs , method , max_workers , ignore_error)

