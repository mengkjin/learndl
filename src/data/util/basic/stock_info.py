import numpy as np
import pandas as pd
from typing import Any

from src.basic import PATH , CONF , CALENDAR , TradeDate
from src.func.singleton import singleton

@singleton
class InfoDataAccess:
    def __init__(self) -> None:
        self.desc = PATH.db_load('information_ts' , 'description') 
        self.desc['list_dt'] = np.maximum(self.desc['list_dt'] , CALENDAR.calendar_start())

        self.cname = PATH.db_load('information_ts' , 'change_name') 
        self.cname = self.cname[self.cname['secid'] >= 0].sort_values(['secid','ann_date','start_date']).rename(columns={'ann_date':'ann_dt'})

        self.indus_dict = pd.DataFrame(CONF.glob('tushare_indus'))
        self.indus_data = PATH.db_load('information_ts' , 'industry') 

        self.indus_data['indus'] = self.indus_dict.loc[self.indus_data['l2_name'],'indus'].values
        self.indus_data = self.indus_data.sort_values(['secid','in_date'])

    def get_desc(self , date : int | TradeDate | None = None , set_index : bool = True , listed = True , exchange = ['SZSE', 'SSE', 'BSE']):
        desc = self.desc
        if date is not None: 
            desc = desc.loc[(desc['list_dt'] <= int(date)) & (desc['delist_dt'] > int(date))]
        if listed: desc = desc[desc['list_dt'] > 0]
        if exchange: desc = desc[desc['exchange_name'].isin(exchange)]
        if set_index: desc = desc.set_index('secid')
        return desc

    def get_secid(self , date : int | None = None): 
        return self.get_desc(date , set_index=False)['secid'].unique()
    
    def get_st(self , date : int | TradeDate | None = None , reason = ['终止上市', '暂停上市' , 'ST', '*ST']):
        new_cname = self.cname[self.cname['change_reason'].isin(reason)]
        if date is not None: 
            new_cname = self.cname[self.cname['start_date'] <= date].copy().drop_duplicates('secid' , keep = 'last')
        return new_cname.loc[:,['secid','entry_dt','remove_dt','ann_dt']]
    
    def get_list_dt(self , date : int | TradeDate | None = None , offset = 0):
        desc = self.get_desc(date)
        if offset != 0: desc['list_dt'] = CALENDAR.td_array(desc['list_dt'] , offset)
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

    def add_indus(self , df : pd.DataFrame , date : int | TradeDate | None = None , na_industry_as : Any = None):
        df = df.join(self.get_indus(date) , on = 'secid' , how = 'left')
        if na_industry_as is not None: df['indus'] = df['indus'].fillna(na_industry_as)
        return df

    def get_listed_mask(self , df : pd.DataFrame , list_dt_offset = 21 , reference_date : int | TradeDate | None = None):
        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(df.columns.values).fillna(99991231).astype(int).reset_index()['list_dt']
        df_date = pd.DataFrame(np.tile(df.index.values[:, None], (1, df.shape[1])), index=df.index, columns=df.columns)
        return np.where(list_dt > df_date , np.nan , 0)
    
    def mask_list_dt(self , df : pd.DataFrame , mask : bool = True , list_dt_offset : int = 21 , reference_date : int | None = None):
        if not mask: return df
        pivoted = df.columns.name == 'secid' and df.index.name == 'date'
        date_values  = df.index.values   if pivoted else df.index.get_level_values('date').values
        secid_values = df.columns.values if pivoted else df.index.get_level_values('secid').values

        if reference_date is None: reference_date = date_values.max()

        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(secid_values).fillna(99991231).astype(int).values
        if pivoted: list_dt = list_dt.T
        
        df_date = pd.DataFrame(np.tile(date_values[:, None], (1, df.shape[1])))
        list_dt_mask = pd.DataFrame(np.where(df_date < list_dt , np.nan , 0) , index = df.index , columns = df.columns)

        return df + list_dt_mask
    
INFO = InfoDataAccess()