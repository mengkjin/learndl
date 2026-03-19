from types import NoneType
import numpy as np
import pandas as pd

from typing import Any , Generator

from .trade_date import TradeDate

class Dates:
    """
    Dates util , takes possible input types and returns a list representation of dates, can convert to np.ndarray
    accepts:
    0. empty input: 
       - empty dates[]
    1. only 1 input arg:
       - int or TradeDate or string of digits
       - list / np.ndarray / pd.Series)
    2. multiple input args:
       - 2 and only 2 ints or TradeDates or strings of digits , start ~ end , None can be used as placeholder
       - any number of list / np.ndarray / pd.Series , will be concatenated and uniqued
    """
    MIN_DATE = 19900101
    MAX_DATE = 99991231
    def __init__(self , *args : Any):
        self._start_date = None
        self._end_date = None
        self._dates = None
        if len(args) == 0:
            self.dates = []
        elif len(args) == 1:
            assert args[0] is not None , 'None is not allowed as a single input argument'
            if isinstance(args[0] , (int , TradeDate , str)):
                self.dates = [int(args[0])]
            else:
                self.dates = args[0]
        else:
            if len(args) == 2 and isinstance(args[0] , (int , TradeDate , str , NoneType)) and isinstance(args[1] , (int , TradeDate , str , NoneType)):
                self.start_date = args[0]
                self.end_date = args[1]
            else:
                self.dates = np.unique(np.concatenate([self.to_dates(arg) for arg in args]))

    @property
    def start_date(self) -> int:
        return self._start_date or self.MIN_DATE

    @start_date.setter
    def start_date(self , value : int | TradeDate | str | None):
        if value is not None:
            value = int(value)
        self._start_date = value
    
    @property
    def end_date(self) -> int:
        return self._end_date or self.MAX_DATE

    @end_date.setter
    def end_date(self , value : int | TradeDate | str | None):
        if value is not None:
            value = int(value)
        self._end_date = value
    
    @property
    def dates(self) -> np.ndarray:
        return np.array([], dtype=int) if self._dates is None else self._dates

    @dates.setter
    def dates(self , value : 'Dates | np.ndarray | pd.Series | list | None'):
        self._dates = None if value is None else self.to_dates(value)

    def get_dates(self) -> np.ndarray:
        if (not hasattr(self , '_dates') or self._dates is None) and (self.start_date is not None or self.end_date is not None):
            from src.proj.calendar import CALENDAR
            self.dates = CALENDAR.td_within(self.start_date , self.end_date)
        return self.dates

    def __len__(self) -> int:
        return len(self.get_dates())

    def __repr__(self) -> str:
        return self.format_str()

    def __iter__(self) -> Generator[int , None , None]:
        for date in self.get_dates():
            yield date

    def contains(self , date : int | TradeDate | str) -> bool:
        return date in self.get_dates()

    @property
    def empty(self) -> bool:
        if self._dates is None:
            return self.start_date > self.end_date
        else:
            return len(self._dates) == 0

    def diffs(self , other : 'Dates | np.ndarray | pd.Series | list') -> np.ndarray:
        return np.setdiff1d(self.get_dates() , self.to_dates(other))

    @classmethod
    def to_dates(cls , input : 'Dates | np.ndarray | pd.Series | list') -> np.ndarray:
        if isinstance(input , Dates):
            return input.get_dates()
        elif isinstance(input , np.ndarray | pd.Series | list):
            return np.array(input, dtype=int)
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

    def format_str(self) -> str:
        if self.empty:
            return 'Empty Dates'
        elif (dates := self._dates) is not None:
            return f'{dates.min()}~{dates.max()}({len(dates)}days)' if len(dates) > 1 else str(dates[0])
        else:
            return '~'.join([str(date) for date in [self.start_date , self.end_date] if date is not None])

class Dates2(np.ndarray):
    """
    Dates util , takes possible input types and returns a list representation of dates, can convert to np.ndarray
    accepts:
    0. empty input: 
       - empty dates[]
    1. only 1 input arg:
       - int or TradeDate or string of digits
       - list / np.ndarray / pd.Series)
    2. multiple input args:
       - 2 and only 2 ints or TradeDates or strings of digits , start ~ end , None can be used as placeholder
       - any number of list / np.ndarray / pd.Series , will be concatenated and uniqued
    """
    MIN_DATE = 19900101
    MAX_DATE = 99991231
    def __init__(self , *args : Any):
        self._start_date = None
        self._end_date = None
        self._dates = None
        if len(args) == 0:
            self.dates = []
        elif len(args) == 1:
            assert args[0] is not None , 'None is not allowed as a single input argument'
            if isinstance(args[0] , (int , TradeDate , str)):
                self.dates = [int(args[0])]
            else:
                self.dates = args[0]
        else:
            if len(args) == 2 and isinstance(args[0] , (int , TradeDate , str , NoneType)) and isinstance(args[1] , (int , TradeDate , str , NoneType)):
                self.start_date = args[0]
                self.end_date = args[1]
            else:
                self.dates = np.unique(np.concatenate([self.to_dates(arg) for arg in args]))

    @property
    def start_date(self) -> int:
        return self._start_date or self.MIN_DATE

    @start_date.setter
    def start_date(self , value : int | TradeDate | str | None):
        if value is not None:
            value = int(value)
        self._start_date = value
    
    @property
    def end_date(self) -> int:
        return self._end_date or self.MAX_DATE

    @end_date.setter
    def end_date(self , value : int | TradeDate | str | None):
        if value is not None:
            value = int(value)
        self._end_date = value
    
    @property
    def dates(self) -> np.ndarray:
        return np.array([], dtype=int) if self._dates is None else self._dates

    @dates.setter
    def dates(self , value : 'Dates | np.ndarray | pd.Series | list | None'):
        self._dates = None if value is None else self.to_dates(value)

    def get_dates(self) -> np.ndarray:
        if (not hasattr(self , '_dates') or self._dates is None) and (self.start_date is not None or self.end_date is not None):
            from src.proj.calendar import CALENDAR
            self.dates = CALENDAR.td_within(self.start_date , self.end_date)
        return self.dates

    def __len__(self) -> int:
        return len(self.get_dates())

    def __repr__(self) -> str:
        return self.format_str()

    def __iter__(self) -> Generator[int , None , None]:
        for date in self.get_dates():
            yield date

    def contains(self , date : int | TradeDate | str) -> bool:
        return date in self.get_dates()

    @property
    def empty(self) -> bool:
        if self._dates is None:
            return self.start_date > self.end_date
        else:
            return len(self._dates) == 0

    def diffs(self , other : 'Dates | np.ndarray | pd.Series | list') -> np.ndarray:
        return np.setdiff1d(self.get_dates() , self.to_dates(other))

    @classmethod
    def to_dates(cls , input : 'Dates | np.ndarray | pd.Series | list') -> np.ndarray:
        if isinstance(input , Dates):
            return input.get_dates()
        elif isinstance(input , np.ndarray | pd.Series | list):
            return np.array(input, dtype=int)
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

    def format_str(self) -> str:
        if self.empty:
            return 'Empty Dates'
        elif (dates := self._dates) is not None:
            return f'{dates.min()}~{dates.max()}({len(dates)}days)' if len(dates) > 1 else str(dates[0])
        else:
            return '~'.join([str(date) for date in [self.start_date , self.end_date] if date is not None])