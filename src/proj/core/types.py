"""
Basic types alias for the project for type hints
"""
from __future__ import annotations
import numpy as np
import enum
from typing import (
    Any  , TypeAlias  , Self , TypeVar)
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from .trade_date import TradeDate

__all__ = [
    'strPath' , 'strPaths' , 'intNums' , 
    'intDate' , 'intDateNone' , 'intDates' , 'intDatesNone' ,
    'ArrayLike' ,
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

class StrEnum(enum.StrEnum):
    """Custom string enum"""

    @classmethod
    def values(cls) -> tuple[str,...]:
        return tuple(cls.value for cls in cls)

    @classmethod
    def ensure_list(cls , x : Any | Iterable) -> list[Self]:
        if x in cls:
            ret = [x]
        elif isinstance(x , list):
            ret = x
        else:
            ret = list(cls)
        ret = [cls(i) for i in ret]
        return ret