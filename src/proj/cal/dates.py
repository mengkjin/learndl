"""
Trading-calendar utilities: natural dates (cd) vs trading dates (td), ranges, diffs, and quarter helpers.

Uses ``BasicCalendar`` (``BC``) for fast ndarray-backed lookups. Prefer ``CALENDAR`` and ``TradeDate``
from this package for all research code.
"""

from __future__ import annotations
import numpy as np

from copy import deepcopy
from functools import cached_property
from typing import Any, Self , overload , Sequence , Iterator , Iterable

from src.proj.core import lit , as_int_array

from .basic import intDate , intDateNone , intDates , BC

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
            - dates : list / np.ndarray / pd.Series / Dates , directly converted to np.ndarray
        2. exact 2 inputs: start , end
        3. exact 3 inputs: dates , start , end
            - will use start and end to slice the dates
    """
    @overload
    def __init__(
        self , start : intDateNone , end : intDateNone , / , * , 
        type : lit.intDateType | None = None , updated = False , **kwargs):
        """input with a start and end date"""
    @overload
    def __init__(
        self , dates : intDates | Dates | None , / , * ,
        type : lit.intDateType | None = None , updated = False , **kwargs):
        """input with a single date or a sequence of dates"""
    @overload
    def __init__(
        self , dates : intDates | Dates | None , 
        start : intDateNone , end : intDateNone , / , * ,
        type : lit.intDateType | None = None , updated = False , **kwargs):
        """input with a sequence of dates, start and end date"""
    @overload
    def __init__(self , / , * , type : lit.intDateType | None = None , updated = False , **kwargs):
        """input with no arguments"""
    def __init__(self , *args , type : lit.intDateType | None = None , updated = False , **kwargs):
        """input with a single date or a sequence of dates"""
        self._updated = updated
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
            self._raw_dates = as_int_array(raw_dates)
        return self

    @cached_property
    def dates(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        """
        Return the actual dates array.
        """
        if self._raw_dates is None and self._start is None and self._end is None:
            dates = np.array([] , dtype = int)
        elif self._raw_dates is None :
            from src.proj.cal.cal import CALENDAR
            dates = CALENDAR.range(self._start , self._end , self._type , updated = self._updated)
        else:
            dates = self._raw_dates
            dates = dates[dates >= int(self._start or 0)]
            dates = dates[dates <= int(self._end or 99991231)]
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
        """return a new Dates object with the sliced dates"""
    @overload
    def __getitem__(self , item : np.ndarray) -> Dates:
        """return a new Dates object with the sliced dates"""
    @overload
    def __getitem__(self , item : Iterable) -> Dates:
        """get a single date or a slice of dates"""
    def __getitem__(self , item : int | slice | np.ndarray | Any) -> int | Dates:
        """get a single date or a slice of dates"""
        if isinstance(item , int):
            return self.dates[item]
        elif isinstance(item , slice):
            return Dates(self.dates[item])
        elif isinstance(item , np.ndarray):
            return Dates(self.dates[item])
        elif isinstance(item , Iterable):
            return Dates(np.atleast_1d(np.asarray(item)))
        else:
            raise ValueError(f"Invalid item: {item}")

    def __iter__(self) -> Iterator[int]:
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

    @property
    def size(self) -> int:
        """
        Return the number of dates in the array.
        """
        return len(self)

    @property
    def empty(self) -> bool:
        """no dates in the array"""
        return self.size == 0

    @property
    def min(self) -> int:
        """
        Return the minimum date in the array. if the array is empty, return 99991231
        """
        if self.empty:
            return 99991231
        return min(self)

    @property
    def max(self) -> int:
        """
        Return the maximum date in the array. if the array is empty, return 19000101
        """
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
        """
        Copy the dates array.
        """
        return deepcopy(self)

    def diff(self , other : intDates | Dates | None , inplace : bool = False) -> Self:
        """
        Calculate the difference between two arrays of dates.
        """
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.setdiff1d(self.dates , Dates(other).dates)
        return self

    def union(self , other : intDates | Dates | None , inplace : bool = False) -> Self:
        """
        Calculate the union of two arrays of dates.
        """
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.union1d(self.dates , Dates(other).dates)
        return self

    def intersect(self , other : intDates | Dates | None , inplace : bool = False) -> Self:
        """
        Calculate the intersection of two arrays of dates.
        """
        if not inplace:
            self = self.copy()
        if other is not None:
            self.dates = np.intersect1d(self.dates , Dates(other).dates)
        return self

    def slice(self , start : intDateNone = None , end : intDateNone = None , step : int = 1 , inplace : bool = False) -> Self:
        """
        Slice the dates within a given range.
        """
        if not inplace:
            self = self.copy()
        if start is not None:
            self.dates = self.dates[self.dates >= int(start)]
        if end is not None:
            self.dates = self.dates[self.dates <= int(end)]
        if step > 1:
            self.dates = self.dates[::step]
        return self

    def filter(self , year : int | None = None) -> Self:
        """
        Filter the dates by year or other conditions.
        """
        if year is not None:
            self.dates = self.dates[self.dates // 10000 == year]
        return self

    def offset(self, offset: int = 0, type: lit.intDateType = 'td' , backward=True , inplace: bool = True) -> Dates:
        """Convert multiple natural dates to trading dates (or 'td_forward'); optionally offset by 'offset' steps."""
        if self.empty:
            return self
        if not inplace:
            self = self.copy()
        self.dates = BC.offset_np(self.dates, offset, type, backward)
        return self

    def to_td(self , backward=True , inplace: bool = True) -> Dates:
        """Convert the dates to trading dates."""
        return self.offset(offset = 0, type = 'td', backward = backward, inplace = inplace)

    def segments(
        self, max_segment_len : int = 60 , require_consecutive : lit.intDateType | None = None , 
    ) -> list[Dates]:
        """
        return the segments of dates within the max_segment_len 
        if require_consecutive is not None, the segments must be consecutive,
        so breaking dates will the seperated into different segments
        """
        segs = []
        if self.empty:
            return segs
        if require_consecutive is None:
            starts = np.arange(0 , len(self.dates) , max_segment_len)
            ends = np.concatenate([starts[1:] , [len(self.dates)]])
            for start , end in zip(starts, ends):
                segs.append(Dates(self.dates[start:end]))
        else:
            full_dates = BC.range(self.min , self.max , require_consecutive)
            pos = np.sum(self.dates[:,None] - full_dates[None,:] > 0 , axis=1)
            ends = np.concatenate([np.arange(len(pos))[np.diff(pos , prepend = pos[0]) > 1] , [len(pos)]])
            starts = np.concatenate([[0],ends[:-1]])
            for start , end in zip(starts, ends):
                segs.extend(Dates(self.dates[start:end]).segments(max_segment_len , require_consecutive = None))
        return segs