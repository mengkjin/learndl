"""
Trading-calendar utilities: natural dates (cd) vs trading dates (td), ranges, diffs, and quarter helpers.

Uses ``BasicCalendar`` (``BC``) for fast ndarray-backed lookups. Prefer ``CALENDAR`` and ``TradeDate``
from this package for all research code.
"""

from __future__ import annotations
import numpy as np

from copy import deepcopy
from functools import cached_property
from typing import Any, Self , overload , Sequence

from src.proj.core import lit

from .basic import intDate , intDateNone , intDates

__all__ = ['Dates' , 'intDate' , 'intDateNone' , 'intDates']
    
class Dates(Sequence[int]):
    """
    An integer date array view.
    inputs:
        0. empty input:
            - empty dates[]
        1. only 1 input arg:
            - None : length 0 array
            - date : int / TradeDate / string of digits , length 1 array
            - tuple : (start, end) or (start, end, step) , treat as 2 / 3 inputs
            - dates : list / np.ndarray / pd.Series / Dates2 , directly converted to np.ndarray
        2. exact 2 inputs: start , end
        3. exact 3 inputs: dates , start , end
            - will use start and end to slice the dates
    """
    @overload
    def __init__(
        self , start : intDate , end : intDateNone , / , * , 
        type : lit.intDateType | None = None , **kwargs):
        """input with a start and end date"""
    @overload
    def __init__(
        self , dates : intDates | Dates | None , / , * ,
        type : lit.intDateType | None = None , **kwargs):
        """input with a single date or a sequence of dates"""
    @overload
    def __init__(
        self , dates : intDates | Dates | None , 
        start : intDate , end : intDateNone , / , * ,
        type : lit.intDateType | None = None , **kwargs):
        """input with a sequence of dates, start and end date"""
    @overload
    def __init__(self , / , * , type : lit.intDateType | None = None , **kwargs):
        """input with no arguments"""
    def __init__(self , *args , type : lit.intDateType | None = None , **kwargs):
        """input with a single date or a sequence of dates"""
        self._raw_dates = None
        self._start , self._end = None , None
        if len(args) == 0:
            ...
        elif len(args) == 1:
            self._feed_raw_dates(args[0])
        elif len(args) == 2:
            self._start , self._end = args
        elif len(args) == 3:
            self._feed_raw_dates(args[0])
            self._start , self._end = args[1:]
        else:
            raise ValueError(f"Invalid input: {args}")
        self._type : lit.intDateType = type or getattr(self , '_type' , None) or 'td'

    def _feed_raw_dates(self , raw_dates : intDates | Dates | None) -> Self:
        if raw_dates is None:
            self._raw_dates = None
        elif isinstance(raw_dates , Dates):
            self._raw_dates = raw_dates.dates
            self._type = raw_dates._type
        else:
            self._raw_dates = np.atleast_1d(np.array(raw_dates, dtype=int))
        return self

    @cached_property
    def dates(self) -> np.ndarray[int, Any]:
        from src.proj.cal.cal import CALENDAR
        if self._raw_dates is None:
            if self._start is None and self._end is None:
                dates = np.array([] , dtype = int)
            else:
                dates = CALENDAR.range(self._start , self._end , self._type)
        else:
            dates = np.atleast_1d(np.array(self._raw_dates , dtype = int))
            if self._start is not None:
                dates = dates[dates >= int(self._start)]
            if self._end is not None:
                dates = dates[dates <= int(self._end)]
        dates = np.unique(dates)
        return dates

    def __bool__(self) -> bool:
        return not self.empty

    def __len__(self):
        return len(self.dates)

    @overload
    def __getitem__(self , item : int) -> int:
        """get a single date"""
    @overload
    def __getitem__(self , item : slice) -> Dates:
        """return a new Dates2 object with the sliced dates"""
    def __getitem__(self , item : int | slice) -> int | Dates:
        """get a single date or a slice of dates"""
        if isinstance(item , int):
            return self.dates[item]
        elif isinstance(item , slice):
            return Dates(self.dates[item])
        else:
            raise ValueError(f"Invalid item: {item}")

    def __iter__(self):
        return iter(self.dates)

    def __array__(self , dtype = None):
        return np.array(self.dates , dtype = dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.format_str()})"

    def __str__(self) -> str:
        return self.format_str()

    def __contains__(self , item : int) -> bool:
        return item in self.dates

    def __add__(self , other : intDates | Dates | None) -> Self:
        return self.union(other , inplace = False)

    def __sub__(self , other : intDates | Dates | None) -> Self:
        return self.diff(other , inplace = False)

    def with_info(self , info : Any) -> Self:
        self.info = info
        return self

    @property
    def size(self) -> int:
        return len(self)

    @property
    def empty(self) -> bool:
        """no dates in the array"""
        return self.size == 0

    @property
    def min(self) -> int:
        if self.empty:
            return 99991231
        return min(self)

    @property
    def max(self) -> int:
        if self.empty:
            return 19000101
        return max(self)

    def format_str(self) -> str:
        """short text description of empty, single or multiple days."""
        if self.empty:
            return "Empty Dates"
        if len(self) > 1:
            return f"{self.min}~{self.max}({len(self)}days)"
        return str(self[0])

    def copy(self) -> Self:
        return deepcopy(self)

    def diff(self , other : intDates | Dates | None , inplace : bool = True) -> Self:
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.setdiff1d(self.dates , Dates(other).dates)
        return self

    def union(self , other : intDates | Dates | None , inplace : bool = True) -> Self:
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.union1d(self.dates , Dates(other).dates)
        return self

    def intersect(self , other : intDates | Dates | None , inplace : bool = True) -> Self:
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.intersect1d(self.dates , Dates(other).dates)
        return self

    def truncate(self , start : intDateNone = None , end : intDateNone = None , inplace : bool = True) -> Self:
        if not inplace:
            self = self.copy()
        if start is not None:
            self.dates = self.dates[self.dates >= int(start)]
        if end is not None:
            self.dates = self.dates[self.dates <= int(end)]
        return self