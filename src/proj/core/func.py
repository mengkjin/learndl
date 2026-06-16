"""
Most fundamental functions for the project.
"""

from __future__ import annotations
import numpy as np
from typing import Any
from collections.abc import Iterable

__all__ = ['as_int_array' , 'as_float_array' , 'as_str_array']

def _as_array(array_like: Any , dtype : type[int] | type[float] | type[str]) -> np.ndarray:
    """
    Convert an array-like object to a numpy array.
    Supported types: np.ndarray, Iterable, str, TradeDate, Series, Tensor, Dates , int , float
    """
    if array_like is None:
        arr = []
    elif isinstance(array_like, np.ndarray):
        arr = array_like
    elif not isinstance(array_like, Iterable) or isinstance(array_like, str):
        if array_like.__class__.__name__ == 'TradeDate':
            array_like = int(array_like)
        arr = np.atleast_1d(array_like)
    elif array_like.__class__.__name__ == 'Series' and hasattr(array_like, 'to_numpy'):
        arr = array_like.to_numpy() # type: ignore[attr-defined]
    elif array_like.__class__.__name__ == 'Tensor':
        arr = array_like.cpu().numpy() # type: ignore[attr-defined]
    elif array_like.__class__.__name__ == 'Dates' and hasattr(array_like, 'dates'):
        arr = array_like.dates # type: ignore[attr-defined]
    elif isinstance(array_like, Iterable):
        arr = np.atleast_1d(np.asarray(array_like))
    else:
        raise ValueError(f'Invalid dates type: {type(array_like)} {array_like}')
    assert isinstance(arr, np.ndarray), f'dates is not np.ndarray: {type(arr)}'
    return arr.astype(dtype)

def as_int_array(ints: Any) -> np.ndarray[Any, np.dtype[np.int_]]:
    return _as_array(ints, int)

def as_float_array(floats: Any) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    return _as_array(floats, float)

def as_str_array(strings: Any) -> np.ndarray[Any, np.dtype[np.str_]]:
    return _as_array(strings, str)