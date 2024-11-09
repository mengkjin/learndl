import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from .abstract_data_data import DateDataAccess

from ..basic.trade_date import TradeDate
from ....basic import PATH
from ....func.singleton import singleton

@singleton
class FinaDataAccess(DateDataAccess):
    MAX_LEN = 40
    DATA_TYPE_LIST = ['indi']

    def __init__(self) -> None:
        super().__init__()
        self.QE = self.full_quarter_ends()

    def loader_func(self , date , data_type):
        if data_type == 'indi': 
            df = PATH.load_target_file('financial_ts' , 'indicator' , date)
            df['ann_date'] = df['ann_date'].fillna(99991231).astype(int)
            df['end_date'] = df['end_date'].fillna(-1).astype(int)
            return df
        else:
            raise KeyError(data_type)

    def get_indi(self , date , cols = None):
        assert date in self.QE , date
        return self.get_df(date , 'indi' , cols)
    
    @property
    def indi_cols(self):
        if self.data_dict['indi']: 
            date = list(self.data_dict['indi'].keys())[0]
            return self.data_dict['indi'][date].columns.values
        else:
            return self.get_indi(20231231).columns.values
    
    @staticmethod
    def full_quarter_ends(start_year = 1997 , end_year = 2099):
        date_list = np.sort(np.concatenate([np.arange(start_year , end_year) * 10000 + qe for qe in [331,630,930,1231]]))
        return date_list
    
    @staticmethod
    def full_year_ends(start_year = 2000 , end_year = 2099):
        date_list = np.arange(start_year , end_year) * 10000 + 1231
        return date_list
    
    def qtr_ends(self , date , lastn = 1 , year_only = False):
        y = date//10000
        if year_only:
            n_years = max(1 , lastn + 1)
            q_ends = self.full_year_ends(y-n_years,y + 1)
        else:
            n_years = max(1 , lastn // 4 + 1)
            q_ends = self.full_quarter_ends(y-n_years,y + 1)
        return q_ends[q_ends < date]

    def get_acc(self , val : str , date : int , lastn = 1 , stack = True , year_only = False):
        q_ends = self.qtr_ends(date , lastn , year_only = year_only)

        cols = ['secid' , 'ann_date' , 'end_date' , 'update_flag' , val]

        df_acc = pd.concat([self.get_indi(qe , cols) for qe in q_ends])
        df_acc = df_acc[(df_acc['ann_date'] <= date) & df_acc['end_date'].isin(q_ends) & (df_acc['secid'] >= 0)]
        df_acc = df_acc.sort_values('update_flag').drop_duplicates(['secid' , 'end_date'] , keep = 'last')\
            [['secid','end_date',val]].sort_values(['secid','end_date']).set_index('secid')
        if stack:
            df_acc = df_acc.groupby('secid').tail(lastn)
        else:
            df_acc = df_acc.pivot_table(val , 'end_date' , 'secid').sort_index()
        return df_acc
    
    def get_qtr(self , val : str , date : int , lastn = 1 , stack = True):
        df_acc = self.get_acc(val , date , lastn + 4 , stack = False)
        q_ends = df_acc.index.get_level_values('end_date').unique()
        y_starts = np.unique(q_ends // 10000) * 10000
        df_qtr = pd.concat([df_acc , df_acc.reindex(y_starts).fillna(0)]).sort_index().ffill().\
            fillna(0).diff().reindex(q_ends).where(~df_acc.isna() , np.nan)
        if stack:
            df_qtr = df_acc.stack().reset_index().rename(columns={0:val}).\
                sort_values(['secid','end_date']).set_index('secid').groupby('secid').tail(lastn)
        return df_qtr

    def get_ttm(self , val : str , date : int , lastn = 1 , stack = True):
        df_acc = self.get_acc(val , date , lastn + 8 , stack = False)
        q_ends = df_acc.index.get_level_values('end_date').unique()
        y_starts = np.unique(q_ends // 10000) * 10000
        df_qtr = pd.concat([df_acc , df_acc.reindex(y_starts).fillna(0)]).sort_index().ffill().\
            fillna(0).diff().reindex(q_ends)
        df_ttm = df_qtr.rolling(4).sum().where(~df_acc.isna() , np.nan)
        if stack:
            df_ttm = df_ttm.stack().reset_index().rename(columns={0:val}).sort_values(['secid' , 'end_date']).\
                groupby('secid').tail(lastn).set_index('secid').sort_index()
        return df_ttm

FINA = FinaDataAccess()