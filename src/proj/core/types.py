"""
Basic types alias for the project for type hints
"""
from __future__ import annotations
import numpy as np
import enum
from typing import Any  , TypeAlias  , Self , TypeVar
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from .trade_date import TradeDate
from .literals import ALL , ANY , NONE , SELF , RANDOM , EQUAL

__all__ = [
    'strPath' , 'strPaths' , 'intNums' , 
    'intDate' , 'intDateNone' , 'intDates' , 'intDatesNone' ,
    'ArrayLike' , 'IterableLike' , 
    'StrEnum' , 'ALL' , 'ANY' , 'NONE' , 'SELF' , 'RANDOM' , 'EQUAL' ,
    'as_int_array' , 'as_float_array' , 'as_str_array' ,
]

T = TypeVar('T')

strPath : TypeAlias = Path | str
strPaths : TypeAlias = Mapping[int | Any , strPath] | Iterable[strPath]
intNums : TypeAlias = int | list[int] | np.ndarray[Any, np.dtype[np.int_]] | Sequence[int] | range
intDate : TypeAlias = int | TradeDate
intDateNone : TypeAlias = int | TradeDate | None
intDates : TypeAlias = intDate | Iterable[intDate] | np.ndarray
intDatesNone : TypeAlias = intDates | None
ArrayLike : TypeAlias = list[T] | np.ndarray | Sequence[T]
IterableLike : TypeAlias = T | Iterable[T] | ALL

class StrEnum(enum.StrEnum):
    """Custom string enum"""

    @classmethod
    def values(cls) -> tuple[str,...]:
        return tuple(cls.value for cls in cls)

    @classmethod
    def ensure_list(cls , x : IterableLike[Self]) -> list[Self]:
        if x == 'all':
            ret = [x for x in cls]
        elif x in cls:
            ret = [x]
        elif isinstance(x , list):
            ret = x
        else:
            ret = list(cls)
        ret = [cls(i) for i in ret]
        return ret

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