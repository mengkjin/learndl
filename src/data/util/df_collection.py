import threading
import warnings
import numpy as np
import pandas as pd
import polars as pl

from abc import ABC , abstractmethod
from copy import deepcopy
from typing import Any

from src.basic import TradeDate

lock = threading.Lock()

class _df_collection(ABC):
    def __init__(self , max_len : int = -1 , date_key : str | None = None) -> None:
        self.max_len = max_len
        self.dates : list[int] = []
        self.last_added_date = -1
        self.date_key : str = date_key if date_key else 'inner_date_key'

        self.data_frames : dict[int , pd.DataFrame | pl.DataFrame] = {}
        self.long_frame : pd.DataFrame | Any = pd.DataFrame()

    @abstractmethod
    def add_one_day(self , date : int , df : pd.DataFrame) -> None:
        '''add a DataFrame with given (1) date'''

    @abstractmethod
    def get_one_day(self , date : int , field = None , 
                    rename_date_key : str | None = 'date') -> pd.DataFrame | pl.DataFrame:
        '''get a DataFrame with given (1) date , fields and set_index'''

    @abstractmethod
    def get_multiple_days(self , dates : list[int] | np.ndarray , field = None , 
                          rename_date_key : str | None = 'date' , copy = False) -> pd.DataFrame | pl.DataFrame:
        '''get a DataFrame with given (many) dates , fields and set_index'''

    def __repr__(self):
        return f'{self.__class__.__name__}(max_len={self.max_len} , dates num={len(self.dates)})'

    def __bool__(self):
        return bool(self.dates)
    
    def __len__(self):
        return len(self.dates)
    
    def __contains__(self , date : int | TradeDate):
        return int(date) in self.dates
    
    @property
    def last_added_data(self):
        '''return the last added df'''
        with lock:
            return self.get(self.last_added_date)
    
    def date_diffs(self , dates : list[int | TradeDate] | np.ndarray , overwrite = False):
        '''return the difference between given dates and self.dates'''
        dates = np.array([int(d) for d in dates])
        return dates if overwrite else np.setdiff1d(dates , self.dates)
    
    def get(self , date : int | TradeDate , field = None , rename_date_key : str | None = 'date'):
        '''get a DataFrame with given (1) date , fields and set_index'''
        with lock:
            date = int(date)
            df = self.get_one_day(date)
            df = self.reform_df(df , field , rename_date_key = rename_date_key)
            return df

    def gets(self , dates : list[int] | np.ndarray , field = None , rename_date_key : str | None = 'date' , copy = False):
        '''get a DataFrame with given (many) dates , fields and set_index'''
        with lock:
            assert len(dates) <= self.max_len , f'No more than {self.max_len} dates , got {len(dates)}'
            df = self.get_multiple_days(dates)
            df = self.reform_df(df , field , rename_date_key = rename_date_key)
            if copy: 
                df = deepcopy(df)
        return df
    
    def add(self , date : int | TradeDate , df : pd.DataFrame | None):
        with lock:
            if df is not None and date not in self.dates and df is not None:
                date = int(date)
                self.dates.append(date)
                self.last_added_date = date
                self.add_one_day(date , df)

    def truncate(self):
        '''truncate the df collection to the max_len , reorder the dates'''
        with lock:
            if len(self) > self.max_len > 0:
                self.dates = sorted(self.dates)[-self.max_len:]
                [self.data_frames.pop(key) for key in self.data_frames if key not in self.dates]
                if isinstance(self.long_frame , pd.DataFrame) and self.date_key in self.long_frame.columns:
                    self.long_frame = self.long_frame[self.long_frame[self.date_key].isin(self.dates)].copy()
    
    def reform_df(self , df : pd.DataFrame | pl.DataFrame , field = None , rename_date_key = None):
        '''
        reform a DataFrame with given fields and set_index
        rename_date_key : if not None , rename the date_key column to the given name
        '''
        if len(df) > 0 and field is not None: 
            if isinstance(field , str): 
                field = [field]
            assert np.isin(field , df.columns).all() , f'{field} should be in df.columns : {df.columns}'
            if isinstance(df , pd.DataFrame):
                df = pd.DataFrame(columns = pd.Index([self.date_key , *field])).set_index(self.date_key) if df.empty else df.loc[:,field]
            elif isinstance(df , pl.DataFrame):
                if self.date_key not in field: 
                    field.append(self.date_key)
                df = df.select(field)

        if rename_date_key and rename_date_key != self.date_key:
            assert rename_date_key not in df.columns , f'{rename_date_key} should not be in df.columns : {df.columns}'
            if isinstance(df , pd.DataFrame):
                df = df.rename(index = {self.date_key:rename_date_key})
            elif isinstance(df , pl.DataFrame):
                df = df.rename({self.date_key:rename_date_key})

        return df

    def columns(self):
        if not self: 
            return []
        elif isinstance(self.long_frame , pd.DataFrame) and not self.long_frame.empty: 
            return self.long_frame.columns.tolist()
        else: 
            columns = self.data_frames[self.dates[0]].columns
            return columns if isinstance(columns , list) else columns.tolist()

class DFCollection(_df_collection):
    def __init__(self , max_len : int = -1 , date_key : str | None = None) -> None:
        super().__init__(max_len , date_key)
        self.data_frames : dict[int , pd.DataFrame] = {}
        self.long_frame : pd.DataFrame = pd.DataFrame()
        
    def get(self , date : int | TradeDate , field = None , rename_date_key : str | None = 'date') -> pd.DataFrame | Any:
        return super().get(date , field , rename_date_key)

    def gets(self , dates : list[int] | np.ndarray , field = None , rename_date_key : str | None = 'date' , copy = False) -> pd.DataFrame | Any:
        return super().gets(dates , field , rename_date_key , copy)

    def add_one_day(self , date : int , df : pd.DataFrame):
        '''add a DataFrame with given (1) date'''
        df = df.reset_index([i for i in df.index.names if i] , drop = False).\
            assign(**{self.date_key:date}).set_index(self.date_key)
        self.data_frames[date] = df.dropna(how = 'all')
    
    def get_one_day(self , date : int | TradeDate) -> pd.DataFrame:
        '''get a DataFrame with given (1) date , fields and set_index'''
        if date not in self.dates: 
            return pd.DataFrame()
        if date in self.data_frames: 
            df = self.data_frames[int(date)]
        else: 
            df = self.long_frame.loc[date:date]
        return df
    
    def get_multiple_days(self , dates : list[int] | np.ndarray):
        '''get a DataFrame with given (many) dates , fields and set_index'''
        self.to_long_frame()
        df = self.long_frame.loc[min(dates):max(dates),:] # .reset_index(drop = False)
        return df

    def to_long_frame(self):
        # assert np.isin(dates , self.dates).all() , f'all dates should be in self.dates : {np.setdiff1d(dates , self.dates)}'
        dates_to_do = list(self.data_frames.keys())
        if len(dates_to_do) == 0: 
            return
        dfs = [df for df in ([self.long_frame] + [v for v in self.data_frames.values()]) if df is not None and not df.empty]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            self.long_frame = pd.concat(dfs, copy=False).sort_values(self.date_key)
        self.data_frames.clear()

class PLDFCollection(_df_collection):
    def __init__(self , max_len : int = -1 , date_key : str | None = None) -> None:
        super().__init__(max_len , date_key)
        self.data_frames : dict[int , pl.DataFrame] = {}

    def get(self , date : int | TradeDate , field = None , rename_date_key : str | None = 'date') -> pl.DataFrame | Any:
        return super().get(date , field , rename_date_key)

    def gets(self , dates : list[int] | np.ndarray , field = None , rename_date_key : str | None = 'date' , copy = False) -> pl.DataFrame | Any:
        return super().gets(dates , field , rename_date_key , copy)
    
    def get_one_day(self , date : int) -> pl.DataFrame:
        '''get a DataFrame with given (1) date , fields and set_index'''
        if date not in self.dates: 
            return pl.DataFrame()
        pldf = self.data_frames[date]
        return pldf
    
    def get_multiple_days(self , dates : list[int] | np.ndarray):
        '''get a DataFrame with given (many) dates , fields and set_index'''
        pldf = pl.concat([self.data_frames[date] for date in dates] , how = 'vertical')
        return pldf
    
    def add_one_day(self , date : int , df : pd.DataFrame):
        pldf = pl.from_pandas(df , include_index = bool([i for i in df.index.names if i]))
        pldf = pldf.with_columns(pl.lit(date).alias(self.date_key))
        self.data_frames[date] = pldf
