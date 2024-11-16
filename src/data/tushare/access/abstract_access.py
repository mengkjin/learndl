import numpy as np
import pandas as pd

from abc import ABC , abstractmethod
from typing import Any

from .stock_info import INFO
from ....basic import CALENDAR , TradeDate

INNER_DATE_KEY = 'inner_date_key'

class DFCollection:
    def __init__(self , max_len : int = -1) -> None:
        self.max_len = max_len
        self.dates : list[int] = []
        self.data_frames : dict[int , pd.DataFrame] = {}
        self.long_frame : pd.DataFrame = pd.DataFrame()
        
    def __repr__(self):
        return f'DFCollection(max_len={self.max_len} , dates num={len(self.dates)})'

    def __bool__(self):
        return bool(self.dates)
    
    def __len__(self):
        return len(self.dates)
    
    def __contains__(self , date : int | TradeDate):
        return int(date) in self.dates
    
    def dates_diff(self , dates : list[int] | np.ndarray , overwrite = False):
        return dates if not overwrite else np.setdiff1d(dates , self.dates)
    
    def get(self , date : int | TradeDate , field = None , set_index : str | list[str] | None = None):
        date = int(date)
        assert date in self.dates , f'{date} is not in {self.dates}'
        if date not in self.data_frames:
            df = self.long_frame[self.long_frame[INNER_DATE_KEY] == date]
        else:
            df = self.data_frames[date]
        if field is not None: df = df.loc[:,field]
        if set_index is not None: df = df.reset_index(drop = df.index.names[0] is None).set_index(set_index)
        return df
    
    def gets(self , dates : list[int] | np.ndarray , field = None , set_index : str | list[str] | None = None):
        assert len(dates) <= self.max_len , f'No more than {self.max_len} dates'

        self.to_long_frame(dates)
        assert np.intersect1d(list(self.data_frames.keys()) , dates).size == 0 , \
            f'self.data_frames.keys should not have overlap with dates: {
                np.intersect1d(list(self.data_frames.keys()) , dates)}'
        df = self.long_frame[self.long_frame[INNER_DATE_KEY].isin(dates)]
        if field is not None: 
            if isinstance(field , str): field = [field]
            if INNER_DATE_KEY not in field: field = [f for f in field] + [INNER_DATE_KEY]
            df = pd.DataFrame(columns = field) if df.empty else df.loc[:,field]
        if set_index is not None: df = df.reset_index(drop = df.index.names[0] is None).set_index(set_index)
        return df

    def last_added(self):
        return self.get(self.dates[-1])

    def add(self , date : int | TradeDate , df : pd.DataFrame | None):
        if date not in self.dates:
            if df is None: df = pd.DataFrame()
            self.dates.append(int(date))
            self.data_frames[int(date)] = df

    def to_long_frame(self , dates : list[int] | np.ndarray):
        assert np.isin(dates , self.dates).all() , f'all dates should be in self.dates : {np.setdiff1d(dates , self.dates)}'
        dates_to_do = np.intersect1d(dates , list(self.data_frames.keys())).tolist()
        if not dates_to_do: return
        df_to_append = pd.concat([self.data_frames.pop(d).assign(**{INNER_DATE_KEY:d}) for d in dates_to_do] , copy = False)
        self.long_frame = pd.concat([self.long_frame , df_to_append] , copy = False).sort_values(INNER_DATE_KEY)

    def columns(self):
        if not self: return None
        elif not self.long_frame.empty: return self.long_frame.columns.tolist()
        else: return self.data_frames[self.dates[0]].columns.tolist()

    def truncate(self):
        if len(self) > self.max_len > 0:
            self.dates = sorted(self.dates)[-self.max_len:]
            [self.data_frames.pop(key) for key in self.data_frames if key not in self.dates]
            self.long_frame = self.long_frame[self.long_frame[INNER_DATE_KEY].isin(self.dates)].copy()

class DateDataAccess(ABC):
    MAX_LEN = -1
    DATA_TYPE_LIST = []

    def __init__(self) -> None:
        self.collections : dict[str , DFCollection] = {data_type:DFCollection() for data_type in self.DATA_TYPE_LIST}
       
    @abstractmethod
    def data_loader(self , date , data_type : str) -> pd.DataFrame:
        '''loader function should return a pd.DataFrame'''

    def truncate(self , data_type : str | None = None , drop_old = True):
        if not drop_old or self.MAX_LEN <= 0: return
        data_type_list = [data_type] if isinstance(data_type , str) else self.DATA_TYPE_LIST

        for data_type in data_type_list:
            self.collections[data_type].truncate()

    def get(self , date: int | TradeDate , data_type : str , field = None , overwrite = False):
        if overwrite or date not in self.collections[data_type]:
            self.collections[data_type].add(date , self.data_loader(date , data_type))
        return self.collections[data_type].get(date , field)

    def gets(self , dates: list[int] | np.ndarray , data_type : str , field = None , overwrite = False):
        for date in self.collections[data_type].dates_diff(dates , overwrite):
            self.collections[data_type].add(date , self.data_loader(date , data_type))
        return self.collections[data_type].gets(dates , field)
    
    def get_specific_data(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                          data_type : str , field : list | str , prev = True , mask = False , pivot = False , drop_old = True):
        dates = CALENDAR.td_array(CALENDAR.td_within(start_dt , end_dt) , -1 if prev else 0)
        remain_field = ['secid'] + ([field] if isinstance(field , str) else list(field))

        df = self.gets(dates , data_type , remain_field)
        df['date'] = CALENDAR.td_array(df[INNER_DATE_KEY] , 1 if prev else 0)

        df = df.set_index(['secid' , 'date'])
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
    