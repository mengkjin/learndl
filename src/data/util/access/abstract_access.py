import threading
import numpy as np
import pandas as pd
import polars as pl

from abc import ABC , abstractmethod
from copy import deepcopy
from typing import Any

from src.basic import CALENDAR , TradeDate
from ..basic import INFO

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
    def add_one_day(self , date : int , df : pd.DataFrame | pl.DataFrame) -> None:
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
            if copy: df = deepcopy(df)
        return df
    
    def add(self , date : int | TradeDate , df : pd.DataFrame | pl.DataFrame | None):
        with lock:
            if date not in self.dates and df is not None and df.shape[0] > 0:
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
                if isinstance(self.long_frame , pd.DataFrame):
                    self.long_frame = self.long_frame[self.long_frame[self.date_key].isin(self.dates)].copy()
    
    def reform_df(self , df : pd.DataFrame | pl.DataFrame , field = None , rename_date_key = None):
        '''
        reform a DataFrame with given fields and set_index
        rename_date_key : if not None , rename the date_key column to the given name
        '''
        if field is not None: 
            if isinstance(field , str): field = [field]
            assert np.isin(field , df.columns).all() , f'{field} should be in df.columns : {df.columns}'
            if isinstance(df , pd.DataFrame):
                df = pd.DataFrame(columns = [self.date_key , *field]).set_index(self.date_key) if df.empty else df.loc[:,field]
            elif isinstance(df , pl.DataFrame):
                if self.date_key not in field: field.append(self.date_key)
                df = df.select(field)

        if rename_date_key and rename_date_key != self.date_key:
            assert rename_date_key not in df.columns , f'{rename_date_key} should not be in df.columns : {df.columns}'
            if isinstance(df , pd.DataFrame):
                df = df.rename(index = {self.date_key:rename_date_key})
            elif isinstance(df , pl.DataFrame):
                df = df.rename({self.date_key:rename_date_key})

        return df

    def columns(self):
        if not self: return None
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
        df = df.reset_index([i for i in df.index.names if i] , drop = False).assign(**{self.date_key:date}).set_index(self.date_key)
        self.data_frames[date] = df
    
    def get_one_day(self , date : int | TradeDate) -> pd.DataFrame:
        '''get a DataFrame with given (1) date , fields and set_index'''
        if date not in self.dates: return pd.DataFrame()
        if date in self.data_frames: df = self.data_frames[date]
        else: df = self.long_frame.loc[date:date]
        return df
    
    def get_multiple_days(self , dates : list[int] | np.ndarray):
        '''get a DataFrame with given (many) dates , fields and set_index'''
        self.to_long_frame()
        df = self.long_frame.loc[min(dates):max(dates),:] # .reset_index(drop = False)
        return df

    def to_long_frame(self):
        # assert np.isin(dates , self.dates).all() , f'all dates should be in self.dates : {np.setdiff1d(dates , self.dates)}'
        dates_to_do = list(self.data_frames.keys())
        if len(dates_to_do) == 0: return
        dfs = [df for df in [self.long_frame] + [v for v in self.data_frames.values()] if df is not None and not df.empty]
        if dfs: 
            self.long_frame = pd.concat(dfs , copy = False).sort_values(self.date_key)
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
        if date not in self.dates: return pl.DataFrame()
        df = self.data_frames[date]
        return df
    
    def get_multiple_days(self , dates : list[int] | np.ndarray):
        '''get a DataFrame with given (many) dates , fields and set_index'''
        df = pl.concat([self.data_frames[date] for date in dates] , how = 'vertical')
        return df
    
    def add_one_day(self , date : int , df : pd.DataFrame | pl.DataFrame):
        if isinstance(df , pd.DataFrame): df = pl.from_pandas(df , include_index = bool([i for i in df.index.names if i]))
        df = df.with_columns(pl.lit(date).alias(self.date_key))
        self.data_frames[date] = df

class DateDataAccess(ABC):
    MAX_LEN = -1
    DATA_TYPE_LIST = []
    PL_DATA_TYPE_LIST = []
    DATE_KEY = 'date'

    def __init__(self) -> None:
        self.collections : dict[str , DFCollection] = {
            data_type : DFCollection(self.MAX_LEN , self.DATE_KEY) 
            for data_type in self.DATA_TYPE_LIST}
        self.pl_collections : dict[str , PLDFCollection] = {
            data_type : PLDFCollection(self.MAX_LEN , self.DATE_KEY) 
            for data_type in self.PL_DATA_TYPE_LIST}
       
    @abstractmethod
    def data_loader(self , date , data_type : str) -> pd.DataFrame:
        '''loader function should return a pd.DataFrame'''

    def truncate(self , data_type : str | None = None):
        data_type_list = [data_type] if isinstance(data_type , str) else self.DATA_TYPE_LIST + self.PL_DATA_TYPE_LIST
        for data_type in data_type_list:
            if data_type in self.collections: self.collections[data_type].truncate()
            if data_type in self.pl_collections: self.pl_collections[data_type].truncate()

    def get(self , date: int | TradeDate , data_type : str , field = None , overwrite = False , rename_date_key = None):
        if overwrite or int(date) not in self.collections[data_type]:
            self.collections[data_type].add(date , self.data_loader(date , data_type))
        return self.collections[data_type].get(date , field , rename_date_key = rename_date_key)

    def gets(self , dates: list[int | TradeDate] | np.ndarray , data_type : str , field = None , overwrite = False , rename_date_key = None):
        for date in self.collections[data_type].date_diffs(dates , overwrite):
            self.collections[data_type].add(date , self.data_loader(date , data_type))
        return self.collections[data_type].gets(dates , field , rename_date_key = rename_date_key)
    
    def get_pl(self , date: int | TradeDate , data_type : str , field = None , overwrite = False , rename_date_key = None):
        if overwrite or int(date) not in self.pl_collections[data_type]:
            self.pl_collections[data_type].add(date , self.data_loader(date , data_type))
        return self.pl_collections[data_type].get(date , field , rename_date_key = rename_date_key)

    def gets_pl(self , dates: list[int] | np.ndarray , data_type : str , field = None , overwrite = False , rename_date_key = None):
        for date in self.pl_collections[data_type].date_diffs(dates , overwrite):
            self.pl_collections[data_type].add(date , self.data_loader(date , data_type))
        return self.pl_collections[data_type].gets(dates , field , rename_date_key = rename_date_key)
    
    def get_specific_data(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                          data_type : str , field : list | str , prev = True , mask = False , pivot = False , drop_old = True , 
                          date_step = 1):
        dates = CALENDAR.td_array(CALENDAR.td_within(start_dt , end_dt , date_step) , -1 if prev else 0)
        remain_field = ['secid'] + ([field] if isinstance(field , str) else list(field))

        df = self.gets(dates , data_type , remain_field , rename_date_key = 'date')
        if prev: 
            df = df.reset_index(drop = False)
            df['date'] = CALENDAR.td_array(df['date'] , 1)
            df = df.set_index('date')
        df = df.set_index('secid' , append = True).sort_index()

        if mask:  df = INFO.mask_list_dt(df)
        if pivot: df = df.pivot_table(field , 'date' , 'secid')
        if drop_old: self.truncate(data_type)
        
        return df
    
    @staticmethod
    def mask_min_finite(df : pd.DataFrame | Any , min_finite_ratio = 0.25):
        if min_finite_ratio <= 0: return df
        assert min_finite_ratio <= 1 , f'min_finite_ratio must be less than or equal to 1 , got {min_finite_ratio}'
        pivoted = df.columns.name == 'secid' and df.index.name == 'date'
        if pivoted:
            secid_values = df.columns.values
            min_finite_mask = pd.Series(np.where(np.isfinite(df).sum() < len(df) * min_finite_ratio , np.nan , 0) , index = secid_values)
        else:
            secid_values = df.index.get_level_values('secid').values
            min_finite_mask = df.isna().groupby('secid').sum() > df.groupby('secid').count() * min_finite_ratio
            min_finite_mask = np.where(min_finite_mask.reindex(secid_values).fillna(False).values , np.nan , 0)

        return df + min_finite_mask
    