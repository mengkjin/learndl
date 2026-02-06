from typing import Callable , Iterable
import os
from src.proj.log import Logger

__all__ = ['FilteredIterable' , 'TempFile']

class FilteredIterable:
    def __init__(self, iterable, condition : Callable | Iterable | None = None , **kwargs):
        self.iterable  = iter(iterable)
        if condition is None:
            self.condition = lambda x: True
        elif callable(condition):
            self.condition = condition
        else:
            self.condition = iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: 
                return item

class TempFile:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def __enter__(self):
        return self.file_name

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            os.remove(self.file_name)
        except Exception as e:
            Logger.error(f'Failed to remove temp file: {e}')