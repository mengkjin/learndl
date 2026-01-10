from typing import Callable , Iterable

__all__ = ['FilteredIterable']

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