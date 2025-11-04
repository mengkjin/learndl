import numpy as np
import pandas as pd

from abc import ABC , abstractmethod
from typing import Any

from src.basic import CALENDAR , TradeDate
from src.data.util import INFO , DFCollection , PLDFCollection

class DateDataAccess(ABC):
    MAX_LEN = -1
    DATA_TYPE_LIST = []     # pandas DataFrame
    PL_DATA_TYPE_LIST = []   # polars DataFrame (large size data)
    DATE_KEY = 'date'

    def __init__(self) -> None:
        self.collections : dict[str , DFCollection] = {
            data_type : DFCollection(self.MAX_LEN , self.DATE_KEY) 
            for data_type in self.DATA_TYPE_LIST}
        self.pl_collections : dict[str , PLDFCollection] = {
            data_type : PLDFCollection(self.MAX_LEN , self.DATE_KEY) 
            for data_type in self.PL_DATA_TYPE_LIST}
       
    @abstractmethod
    def data_loader(self , date , data_type : str) -> pd.DataFrame | None:
        '''
        loader function should return a pd.DataFrame or None
        if None is returned , the data will not be added to the collection
        if DataFrame is returned , it will be added to the collection , even if it is empty
        '''

    def truncate(self , data_type : str | None = None):
        data_type_list = [data_type] if isinstance(data_type , str) else self.DATA_TYPE_LIST + self.PL_DATA_TYPE_LIST
        for data_type in data_type_list:
            if data_type in self.collections: 
                self.collections[data_type].truncate()
            if data_type in self.pl_collections: 
                self.pl_collections[data_type].truncate()

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
                          data_type : str , field : list | str | None , prev = True , mask = False , pivot = False , drop_old = True , 
                          date_step = 1):
        dates = CALENDAR.td_array(CALENDAR.td_within(start_dt , end_dt , date_step) , -1 if prev else 0)
        if field is not None:
            remain_field = ['secid'] + ([field] if isinstance(field , str) else list(field))
        else:
            remain_field = None

        df = self.gets(dates , data_type , remain_field , rename_date_key = 'date')
        if prev: 
            df = df.reset_index(drop = False)
            df['date'] = CALENDAR.td_array(df['date'] , 1)
            df = df.set_index('date')
        df = df.set_index('secid' , append = True).sort_index()

        if mask:  
            df = INFO.mask_list_dt(df)
        if pivot: 
            df = df.pivot_table(field , 'date' , 'secid')
        if drop_old: 
            self.truncate(data_type)
        
        return df
    
    @staticmethod
    def mask_min_finite(df : pd.DataFrame | Any , min_finite_ratio = 0.25):
        if min_finite_ratio <= 0: 
            return df
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
    