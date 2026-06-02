from __future__ import annotations
from typing import Callable , Iterable

__all__ = ['FilteredIterable']

class FilteredIterable:
    """Iterator that yields only items passing a callable or parallel boolean stream."""

    def __init__(self, iterable, condition : Callable | Iterable | None = None , **kwargs):
        """If ``condition`` is iterable, zip with items; if callable, filter by predicate."""
        self.iterable  = iter(iterable)
        if condition is None:
            self.condition = lambda x: True
        elif callable(condition):
            self.condition = condition
        else:
            self.condition = iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        """Return self as iterator."""
        return self
    def __next__(self):
        """Skip items until ``condition`` is truthy."""
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: 
                return item