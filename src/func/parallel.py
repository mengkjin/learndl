import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any , Callable , Literal , Mapping

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

class FuncCall:
    INPUT_TYPE = Callable | tuple[Callable , Iterable | None] | tuple[Callable , list | tuple | None , dict | None]

    def __init__(self , key : Any , func_input : INPUT_TYPE , result : dict , catch_errors : tuple[type[Exception],...] = ()):
        self.key = key
        self.result = result
        self.catch_errors = catch_errors
        self.func , self.args , self.kwargs = self.unwrap(func_input)

    def __repr__(self):
        return f'FuncCall(key={self.key},func={self.func},args={self.args},kwargs={self.kwargs})'

    @staticmethod
    def unwrap(func_input : INPUT_TYPE) -> tuple[Callable , list , dict]:
        args , kwargs = [] , {}
        if isinstance(func_input , Callable):
            func = func_input
        elif isinstance(func_input , tuple):
            func = func_input[0]
            if len(func_input) == 2:
                obj = func_input[1]
                if obj is None:
                    ...
                elif isinstance(obj , dict):
                    kwargs = obj
                elif isinstance(obj , (list,tuple)):
                    args = list(obj)
                else:
                    raise ValueError(f'func_input[1] should be None , dict , list or tuple , but got {type(obj)}')
            elif len(func_input) == 3:
                args , kwargs = list(func_input[1] or []) , func_input[2] or {}
            else:
                raise ValueError(f'func_input should be a tuple of length 2 or 3 , but got {len(func_input)}')
        else:
            raise ValueError(f'func_input should be a Callable or a tuple , but got {type(func_input)}')
        return func , args , kwargs

    def __call__(self) -> Any:
        return self.do()

    def do(self) -> Any:
        try: 
            self.result[self.key] = self.func(*self.args , **self.kwargs)
        except self.catch_errors as e:
            print(f'{self.key} : {self.func}({self.args} , {self.kwargs}) generated an exception: {e}')
        except Exception as e:
            print(f'{self.key} : {self.func}({self.args} , {self.kwargs}) generated an exception: {e}')
            raise e

    def submit(self , pool : ThreadPoolExecutor | ProcessPoolExecutor):
        return pool.submit(self.do)

    @classmethod
    def from_func_calls(cls , inputs : Mapping[Any , INPUT_TYPE] , ignore_error : bool = False) -> tuple[dict[Any , Any] , list['FuncCall']]:
        iterance = inputs.items() if isinstance(inputs , dict) else enumerate(inputs)
        result : dict[Any , Any] = {}
        catch_errors = (Exception , ) if ignore_error else ()
        func_calls : list[FuncCall] = []
        for key , input in iterance:
            func_calls.append(FuncCall(key , input , result , catch_errors = catch_errors))
        return result , func_calls

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
    result_dict : dict , key : Any , 
    func_call : TYPE_FUNC_CALL , 
    catch_errors : tuple[type[Exception],...] =  () , 
) -> Any:
    func , args , kwargs = unwrap_func_call(func_call)
    try: 
        result_dict[key] = func(*args , **kwargs)
    except catch_errors as e:
        print(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}')
    except Exception as e:
        print(f'{key} : {func}({args} , {kwargs}) generated an exception: {e}')
        raise e

def parallel(
    inputs : Mapping[Any , FuncCall.INPUT_TYPE] , 
    method : str | int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
    max_workers = MAX_WORKERS , ignore_error = False
):
    result , func_calls = FuncCall.from_func_calls(inputs , ignore_error = ignore_error)
    method = get_method(method , max_workers)

    if method == 0:
        [func_call() for func_call in func_calls]
    else:
        PoolExecutor = ProcessPoolExecutor if method == 2 else ThreadPoolExecutor
        with PoolExecutor(max_workers=max_workers) as pool:
            futures = [func_call.submit(pool) for func_call in func_calls]
            for future in as_completed(futures):
                future.result()
    return result

def parallels(
    func_calls : Mapping[Any , TYPE_FUNC_CALL] , 
    method : str | int | bool | Literal['forloop' , 'thread' , 'process'] = 'thread' , 
    max_workers = MAX_WORKERS , ignore_error = False
):
    method = get_method(method , max_workers)
    result : dict[Any , Any] = {}
    iterance = func_calls.items() if isinstance(func_calls , dict) else enumerate(func_calls)
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