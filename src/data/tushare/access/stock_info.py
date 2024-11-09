import numpy as np
import pandas as pd

from abc import ABC , abstractmethod

from .calendar import CALENDAR
from ..basic import TradeDate
from ....basic import PATH , CONF
from ....func.singleton import singleton

class DateDataAccess(ABC):
    MAX_LEN = -1
    DATA_TYPE_LIST = []

    def __init__(self) -> None:
        self.data_dict : dict[str , dict[int , pd.DataFrame]] = {data_type : {} for data_type in self.DATA_TYPE_LIST}

    def load_dates(self , dates , drop_old = True):
        for data_type in self.DATA_TYPE_LIST:
            for date in dates:  self.load(date , data_type , False)
            self.len_control(data_type , drop_old)

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

    def len_control(self , data_type : str , drop_old = False):
        if drop_old and len(self.data_dict[data_type]) > self.MAX_LEN:
            drop_keys = sorted(list(self.data_dict[data_type].keys()))[:-self.MAX_LEN]
            [self.data_dict[data_type].__delitem__(key) for key in drop_keys]

    def get_df(self , date: int | TradeDate , data_type : str , cols = None , drop_old = False):
        df = self.load(date , data_type , drop_old)
        if df is not None and cols is not None:  df = df.loc[:,cols]
        return df

@singleton
class InfoDataAccess:
    def __init__(self) -> None:
        self.desc = pd.read_feather(PATH.get_target_path('information_ts' , 'description'))
        self.cname = pd.read_feather(PATH.get_target_path('information_ts' , 'change_name'))
        self.cname = self.cname[self.cname['secid'] >= 0].sort_values(['secid','ann_date','start_date'])

        self.indus_dict = pd.DataFrame(CONF.glob('tushare_indus'))
        self.indus_data = pd.read_feather(PATH.get_target_path('information_ts' , 'industry'))

        self.indus_data['indus'] = self.indus_dict.loc[self.indus_data['l2_name'],'indus'].values
        self.indus_data = self.indus_data.sort_values(['secid','in_date'])

    def get_desc(self , date : int | TradeDate | None = None):
        if date is None: 
            new_desc = self.desc.copy()
        else:
            new_desc = self.desc[(self.desc['list_dt'] <= int(date)) & (self.desc['delist_dt'] > int(date))].copy()
        new_desc['list_dt'] = np.maximum(new_desc['list_dt'] , CALENDAR.calendar_start())
        new_desc = new_desc.set_index('secid')
        return new_desc
    
    def get_list_dt(self , date : int | TradeDate | None = None , offset = 0):
        desc = self.get_desc(date)
        if offset != 0: desc['list_dt'] = CALENDAR.offset(desc['list_dt'] , offset , 't')
        return desc.loc[:,['list_dt']].reset_index().drop_duplicates(subset='secid').set_index('secid')
    
    def get_abnormal(self , date : int | TradeDate | None = None , reason = ['终止上市', '暂停上市' , 'ST', '*ST', ]):
        if date is None: 
            new_cname = self.cname.copy()
        else:
            new_cname = self.cname[self.cname['start_date'] <= date].copy().drop_duplicates('secid' , keep = 'last')
        new_cname = new_cname[new_cname['change_reason'].isin(reason)]
        return new_cname
    
    def get_indus(self , date : int | TradeDate | None = None):
        if date is None: 
            df = self.indus_data.copy()
        else:
            df = self.indus_data[self.indus_data['in_date'] <= int(date)]
        df = df.groupby('secid')[['indus']].last()
        return df

    def get_listed_mask(self , df : pd.DataFrame , list_dt_offset = 21 , reference_date : int | TradeDate | None = None):
        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(df.columns.values).fillna(99991231).astype(int).reset_index()['list_dt']
        df_date = pd.DataFrame(np.tile(df.index.values[:, None], (1, df.shape[1])), index=df.index, columns=df.columns)
        return np.where(list_dt > df_date , np.nan , 0)
    
    def mask_list_dt(self , df : pd.DataFrame , mask : bool = True , list_dt_offset : int = 21 , reference_date : int | None = None):
        if not mask: return df
        pivoted = df.columns.name == 'secid' and df.index.name == 'date'
        date_values  = df.index.values if pivoted else df.index.get_level_values('date').values
        secid_values = df.columns.values if pivoted else df.index.get_level_values('secid').values

        if reference_date is None: reference_date = date_values.max()

        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(secid_values).fillna(99991231).astype(int).values
        if pivoted: list_dt = list_dt.T
        
        df_date = pd.DataFrame(np.tile(date_values[:, None], (1, df.shape[1])))
        list_dt_mask = pd.DataFrame(np.where(df_date < list_dt , np.nan , 0) , index = df.index , columns = df.columns)

        return df + list_dt_mask
    
INFO = InfoDataAccess()