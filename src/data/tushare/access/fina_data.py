import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from .abstract_access import DateDataAccess

from ....basic import CALENDAR , PATH
from ....func.singleton import singleton

QUARTER_ENDS = np.sort(np.concatenate([np.arange(1997 , 2099) * 10000 + qe for qe in [331,630,930,1231]]))
YEAR_ENDS = np.arange(1997 , 2099) * 10000 + 1231

class FDataAccess(DateDataAccess):
    MAX_LEN = 40
    DATA_TYPE_LIST = ['income' , 'cashflow' , 'balance' , 'dividend' , 'disclosure' ,
                      'express' , 'forecast' , 'mainbz' , 'indicator']

    def data_loader(self , date , data_type):
        if data_type in self.DATA_TYPE_LIST: 
            return PATH.db_load('financial_ts' , data_type , date , check_na_cols=False)
        else:
            raise KeyError(data_type)
    @staticmethod
    def full_quarter_ends(start_year = 1997 , end_year = 2099):
        return QUARTER_ENDS[(QUARTER_ENDS >= start_year * 10000) & (QUARTER_ENDS <= end_year * 10000)]
    
    @staticmethod
    def full_year_ends(start_year = 2000 , end_year = 2099):
        return YEAR_ENDS[(YEAR_ENDS >= start_year * 10000) & (YEAR_ENDS <= end_year * 10000)]
    
    def qtr_ends(self , date , n_past = 1 , n_future = 0 , year_only = False):
        if year_only:
            qtr_past = YEAR_ENDS[YEAR_ENDS <= date][-n_past:]
            qtr_future = YEAR_ENDS[YEAR_ENDS >= date][:n_future]
            qtr_ends = np.concatenate([qtr_past , qtr_future])
        else:
            qtr_past = QUARTER_ENDS[QUARTER_ENDS <= date][-n_past:]
            qtr_future = QUARTER_ENDS[QUARTER_ENDS >= date][:n_future]
            qtr_ends = np.concatenate([qtr_past , qtr_future])
        return qtr_ends

    def get_acc_data(self , data_type : str , val : str , date : int , lastn = 1 , stack = True , year_only = False):
        assert data_type in ['income' , 'cashflow' , 'balance' , 'indicator'] , \
            f'invalid data_type: {data_type} , must be one of ' + str(['income' , 'cashflow' , 'balance' , 'indicator'])
        q_ends = self.qtr_ends(date , lastn , year_only = year_only)

        field = ['secid' , 'ann_date' , 'end_date' , 'update_flag' , val]
        
        df_acc = self.gets(q_ends , data_type , field).dropna(subset = ['ann_date' , 'end_date'])
        # df_acc = pd.concat([self.get(qe , data_type , field) for qe in q_ends]).dropna(subset = ['ann_date' , 'end_date'])
        df_acc['ann_date'] = df_acc['ann_date'].astype(int)
        df_acc['end_date'] = df_acc['end_date'].astype(int)
        df_acc = df_acc[(df_acc['ann_date'] <= date) & df_acc['end_date'].isin(q_ends) & (df_acc['secid'] >= 0)]
        df_acc = df_acc.sort_values('update_flag').drop_duplicates(['secid' , 'end_date'] , keep = 'last')\
            [['secid','end_date',val]].sort_values(['secid','end_date']).set_index('secid')
        if stack:
            df_acc = df_acc.groupby('secid').tail(lastn)
        else:
            df_acc = df_acc.pivot_table(val , 'end_date' , 'secid').sort_index()
        return df_acc
    
    def get_qtr_data(self , data_type : str , val : str , date : int , lastn = 1 , stack = True):
        df_acc = self.get_acc_data(data_type , val , date , lastn + 4 , stack = False)
        q_ends = df_acc.index.get_level_values('end_date').unique()
        y_starts = np.unique(q_ends // 10000) * 10000
        df_qtr = pd.concat([df_acc , df_acc.reindex(y_starts).fillna(0)]).sort_index().ffill().\
            fillna(0).diff().reindex(q_ends).where(~df_acc.isna() , np.nan)
        if stack:
            df_qtr = df_acc.stack().reset_index().rename(columns={0:val}).\
                sort_values(['secid','end_date']).set_index('secid').groupby('secid').tail(lastn)
        return df_qtr

    def get_ttm_data(self , data_type : str , val : str , date : int , lastn = 1 , stack = True):
        df_acc = self.get_acc_data(data_type , val , date , lastn + 8 , stack = False)
        q_ends = df_acc.index.get_level_values('end_date').unique()
        y_starts = np.unique(q_ends // 10000) * 10000
        df_qtr = pd.concat([df_acc , df_acc.reindex(y_starts).fillna(0)]).sort_index().ffill().\
            fillna(0).diff().reindex(q_ends)
        df_ttm = df_qtr.rolling(4).sum().where(~df_acc.isna() , np.nan)
        if stack:
            df_ttm = df_ttm.stack().reset_index().rename(columns={0:val}).sort_values(['secid' , 'end_date']).\
                groupby('secid').tail(lastn).set_index('secid').sort_index()
        return df_ttm
    
@singleton
class IndicatorDataAccess(FDataAccess):
    DATA_TYPE_LIST = ['indicator']

    def get_indi(self , date , field = None):
        return self.get(date , 'indicator' , field)
    
    @property
    def fields(self):
        if self.collections['indicator']: 
            return self.collections['indicator'].columns()
        else:
            return self.get_indi(20231231).columns.values

    def get_acc(self , val : str , date : int , lastn = 1 , stack = True , year_only = False):
        return self.get_acc_data('indicator' , val , date , lastn , stack , year_only)
    
    def get_qtr(self , val : str , date : int , lastn = 1 , stack = True):
        return self.get_qtr_data('indicator' , val , date , lastn , stack)

    def get_ttm(self , val : str , date : int , lastn = 1 , stack = True):
        return self.get_ttm_data('indicator' , val , date , lastn , stack)
    
@singleton
class FinancialDataAccess(FDataAccess):
    DATA_TYPE_LIST = ['income' , 'cashflow' , 'balance' , 'dividend' , 'disclosure' ,
                      'express' , 'forecast' , 'mainbz']
    
    def get_ann_dt(self , date , latest_n = 1 , within_days = 365):
        assert latest_n >= 1 , 'latest_n must be positive'
        ann_dt = self.gets(self.qtr_ends(date , latest_n + 5 , 0) , 'income' , ['secid' , 'ann_date']).dropna(subset = ['ann_date'])
        #income_ann_dt = [self.get(qtr_end , 'income' , ['secid' , 'ann_date']) for qtr_end in self.qtr_ends(date , latest_n + 5 , 0)]
        #ann_dt : pd.DataFrame = pd.concat(income_ann_dt)
        ann_dt['ann_date'] = ann_dt['ann_date'].astype(int)
        ann_dt['td_backward'] = CALENDAR.td_array(ann_dt['ann_date'] , backward = True)
        ann_dt['td_forward']  = CALENDAR.td_array(ann_dt['ann_date'] , backward = False)
        ann_dt = ann_dt.loc[ann_dt['td_forward'] <= date , :]
        if within_days > 0:
            ann_dt = ann_dt[ann_dt['td_backward'] >= CALENDAR.cd(date , -within_days)]
        grp = ann_dt.sort_values(['secid' , 'td_backward']).set_index('secid').groupby('secid')
        return grp.last() if latest_n == 1 else grp.tail(latest_n + 1).groupby('secid').first()

FINA = FinancialDataAccess()
INDI = IndicatorDataAccess()