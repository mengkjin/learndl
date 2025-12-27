from typing import Callable , Iterable

__all__ = ['Filtered']

class Filtered:
    def __init__(self, iterable, condition : Callable | Iterable , **kwargs):
        self.iterable  = iter(iterable)
        self.condition = condition if callable(condition) else iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: 
                return item