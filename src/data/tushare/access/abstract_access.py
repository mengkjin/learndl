import numpy as np
import pandas as pd

from abc import ABC , abstractmethod
from typing import Any

from .stock_info import INFO
from ....basic import CALENDAR , TradeDate

class DateDataAccess(ABC):
    MAX_LEN = -1
    DATA_TYPE_LIST = []

    def __init__(self) -> None:
        self.data_dict : dict[str , dict[int , pd.DataFrame]] = {data_type : {} for data_type in self.DATA_TYPE_LIST}

    def load(self, date : int | TradeDate , data_type : str , drop_old = False):
        if data_type not in self.data_dict: raise KeyError(data_type)
        self.loader_to_dict(date , data_type , overwrite = False)
        self.len_control(data_type , drop_old)
        return self.data_dict[data_type][int(date)]

    def loader_to_dict(self , date : int | TradeDate , data_type : str , overwrite = False):
        assert data_type in self.data_dict , f'{data_type} should be in {self.data_dict.keys()}'
        if overwrite or int(date) not in self.data_dict[data_type]:
            self.data_dict[data_type][int(date)] = self.loader_func(int(date) , data_type)

    @abstractmethod
    def loader_func(self , date , data_type : str) -> pd.DataFrame:
        '''loader function should return a pd.DataFrame'''

    def len_control(self , data_type : str | None = None , drop_old = False):
        if not drop_old: return
        data_type_list = [data_type] if isinstance(data_type , str) else self.DATA_TYPE_LIST

        for data_type in data_type_list:
            if len(self.data_dict[data_type]) > self.MAX_LEN:
                drop_keys = sorted(list(self.data_dict[data_type].keys()))[:-self.MAX_LEN]
                [self.data_dict[data_type].__delitem__(key) for key in drop_keys]

    def get_df(self , date: int | TradeDate , data_type : str , field = None , drop_old = False):
        df = self.load(date , data_type , drop_old)
        if df is not None and field is not None:  df = df.loc[:,field]
        return df
    
    def get_specific_data(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                          data_type : str , field : list | str , prev = True , mask = False , pivot = False , drop_old = True):
        assert data_type in self.DATA_TYPE_LIST , f'data_type must be in {self.DATA_TYPE_LIST}'

        dates = CALENDAR.td_within(start_dt , end_dt)
        use_dates = CALENDAR.td_array(dates , -1) if prev else dates

        assert len(dates) <= self.MAX_LEN , f'No more than {self.MAX_LEN} dates between {start_dt} and {end_dt}'
        assert len(dates) == len(use_dates) , 'dates and val_dates should have the same length'

        remain_field = ['secid' , 'date'] + ([field] if isinstance(field , str) else list(field))
        df_list = []
        for d , ud in zip(dates , use_dates):
            df = self.get_df(ud , data_type , remain_field)
            if df is not None: df_list.append(df.assign(date = d))

        if not df_list: 
            df = pd.DataFrame(columns = remain_field).set_index(['secid' , 'date'])
        else:
            df = INFO.mask_list_dt(pd.concat(df_list).set_index(['secid' , 'date']).sort_index() , mask)
        self.len_control(data_type , drop_old)
        if pivot: df = df.pivot_table(field , 'date' , 'secid')
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
    