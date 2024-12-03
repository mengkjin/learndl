import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from src.basic import CALENDAR , PATH
from src.func.singleton import singleton

from .abstract_access import DateDataAccess

QUARTER_ENDS = np.sort(np.concatenate([np.arange(1997 , 2099) * 10000 + qe for qe in [331,630,930,1231]]))
YEAR_ENDS = np.arange(1997 , 2099) * 10000 + 1231

class FDataAccess(DateDataAccess):
    MAX_LEN = 40
    DATE_KEY = 'end_date'
    DATA_TYPE_LIST = ['income' , 'cashflow' , 'balance' , 'dividend' , 'disclosure' ,
                      'express' , 'forecast' , 'mainbz' , 'indicator']
    SINGLE_TYPE : str | Any = None
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    DEFAULT_TTM_METHOD : Literal['sum' , 'avg'] = 'sum'

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
    
    @staticmethod
    def qtr_ends(date , n_past = 1 , n_future = 0 , year_only = False):
        if year_only:
            qtr_past = YEAR_ENDS[YEAR_ENDS <= date][-n_past:]
            qtr_future = YEAR_ENDS[YEAR_ENDS >= date][:n_future]
            qtr_ends = np.concatenate([qtr_past , qtr_future])
        else:
            qtr_past = QUARTER_ENDS[QUARTER_ENDS <= date][-n_past:]
            qtr_future = QUARTER_ENDS[QUARTER_ENDS >= date][:n_future]
            qtr_ends = np.concatenate([qtr_past , qtr_future])
        return qtr_ends
    
    def get_ann_dt(self , date , latest_n = 1 , within_days = 365):
        assert latest_n >= 1 , 'latest_n must be positive'
        assert self.SINGLE_TYPE , 'SINGLE_TYPE must be set'
        ann_dt = self.gets(self.qtr_ends(date , latest_n + 5 , 0) , self.SINGLE_TYPE , ['secid' , 'ann_date']).dropna(subset = ['ann_date'])
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

    def _get_data_acc_hist(self , data_type : str , val : str , date : int , lastn = 1 , stack = True , ffill = False , year_only = False):
        assert data_type in ['income' , 'cashflow' , 'balance' , 'indicator'] , \
            f'invalid data_type: {data_type} , must be one of ' + str(['income' , 'cashflow' , 'balance' , 'indicator'])
        q_ends = self.qtr_ends(date , lastn , year_only = year_only)

        field = ['secid' , 'ann_date' , 'update_flag' , val]
        
        df_acc = self.gets(q_ends , data_type , field , rename_date_key = 'end_date').reset_index(drop = False).dropna(subset = ['ann_date' , 'end_date'])
        # df_acc = pd.concat([self.get(qe , data_type , field) for qe in q_ends]).dropna(subset = ['ann_date' , 'end_date'])
        df_acc['ann_date'] = df_acc['ann_date'].astype(int)
        df_acc['end_date'] = df_acc['end_date'].astype(int)
        df_acc = df_acc[(df_acc['ann_date'] <= date) & df_acc['end_date'].isin(q_ends) & (df_acc['secid'] >= 0)]
        df_acc = df_acc.sort_values('update_flag').drop_duplicates(['secid' , 'end_date'] , keep = 'last')\
            [['secid','end_date',val]].sort_values(['secid','end_date']).set_index('secid')
        
        if ffill: df_acc = df_acc.ffill()

        if stack:
            df_acc = df_acc.groupby('secid').tail(lastn)
        else:
            df_acc = df_acc.pivot_table(val , 'end_date' , 'secid').sort_index()
        return df_acc
    
    def _get_data_qtr_hist(self , data_type : str , val : str , date : int , lastn = 1 , stack = True , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD
        if qtr_method == 'exact':
            return self._get_data_acc_hist(data_type , val , date , lastn , stack , ffill)

        df_acc = self._get_data_acc_hist(data_type , val , date , lastn + 4 , stack = False)
        q_ends = df_acc.index.get_level_values('end_date').unique()
        y_starts = np.unique(q_ends // 10000) * 10000
        df_qtr = pd.concat([df_acc , df_acc.reindex(y_starts).fillna(0)]).sort_index().ffill().fillna(0).\
            diff().reindex(q_ends).where(~df_acc.isna() , np.nan)
        
        if ffill: df_qtr = df_qtr.ffill()
        if stack:
            df_qtr = df_qtr.stack().reset_index().rename(columns={0:val}).\
                sort_values(['secid','end_date']).set_index('secid').groupby('secid').tail(lastn)
        return df_qtr

    def _get_data_ttm_hist(self , data_type : str , val : str , date : int , lastn = 1 , stack = True , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None , 
                           ttm_method : Literal['sum' , 'avg'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD 
        if ttm_method is None: ttm_method = self.DEFAULT_TTM_METHOD
        df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 8 , stack = False , ffill = ffill , qtr_method = qtr_method)
        if ttm_method == 'sum':
            df_ttm = df_qtr.rolling(4).sum()
        elif ttm_method == 'avg':
            df_ttm = df_qtr.rolling(5).mean()
        df_ttm = df_ttm.where(~df_qtr.isna() , np.nan)
        if stack:
            df_ttm = df_ttm.stack().reset_index().rename(columns={0:val}).sort_values(['secid' , 'end_date']).\
                groupby('secid').tail(lastn).set_index('secid').sort_index()
        return df_ttm
    
    def _get_data_acc_latest(self , data_type : str , val : str , date : int , lastn = 1):
        df_acc = self._get_data_acc_hist(data_type , val , date , lastn + 4)
        return df_acc.dropna().groupby('secid').tail(lastn)[val]
    
    def _get_data_qtr_latest(self , data_type : str , val : str , date : int , lastn = 1 , 
                             qtr_method : Literal['diff' , 'exact'] | None = None):
        df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 4 , ffill = False , qtr_method = qtr_method)
        return df_qtr.dropna().groupby('secid').tail(lastn)[val]
    
    def _get_data_ttm_latest(self , data_type : str , val : str , date : int , lastn = 1 ,
                             qtr_method : Literal['diff' , 'exact'] | None = None ,
                             ttm_method : Literal['sum' , 'avg'] | None = None):
        df_ttm = self._get_data_ttm_hist(data_type , val , date , lastn + 4 , ffill = False ,
                                         qtr_method = qtr_method , ttm_method = ttm_method)
        return df_ttm.dropna().groupby('secid').tail(lastn)[val]
    
    @property
    def fields(self):
        assert self.SINGLE_TYPE is not None , 'SINGLE_TYPE must be set'
        if self.collections[self.SINGLE_TYPE]: 
            return self.collections[self.SINGLE_TYPE].columns()
        else:
            return self.get(20231231 , self.SINGLE_TYPE).columns.values

    def acc(self , val : str , date : int , lastn = 1 , stack = True , ffill = False , year_only = False):
        return self._get_data_acc_hist(self.SINGLE_TYPE , val , date , lastn , stack , ffill , year_only)
    
    def qtr(self , val : str , date : int , lastn = 1 , stack = True , ffill = False):
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , stack , ffill , self.DEFAULT_QTR_METHOD)

    def ttm(self , val : str , date : int , lastn = 1 , stack = True , ffill = False):
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , stack , ffill , self.DEFAULT_QTR_METHOD , self.DEFAULT_TTM_METHOD)
    
    def acc_latest(self, val: str, date: int, lastn=1):
        return self._get_data_acc_latest(self.SINGLE_TYPE , val, date, lastn)
    
    def qtr_latest(self, val: str, date: int, lastn=1):
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date, lastn , self.DEFAULT_QTR_METHOD)
    
    def ttm_latest(self, val: str, date: int, lastn=1):
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date, lastn , self.DEFAULT_QTR_METHOD , self.DEFAULT_TTM_METHOD)
    
@singleton
class IndicatorDataAccess(FDataAccess):
    SINGLE_TYPE = 'indicator'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    DEFAULT_TTM_METHOD : Literal['sum' , 'avg'] = 'sum'

    def qtr(self , val : str , date : int , lastn = 1 , stack = True , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , stack , ffill , qtr_method)

    def ttm(self , val : str , date : int , lastn = 1 , stack = True , ffill = False , 
             qtr_method : Literal['diff' , 'exact'] | None = None , ttm_method : Literal['sum' , 'avg'] | None = None):
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , stack , ffill , qtr_method , ttm_method)
    
    def qtr_latest(self, val: str, date: int, lastn=1 , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date, lastn , qtr_method)
    
    def ttm_latest(self, val: str, date: int, lastn=1 , qtr_method : Literal['diff' , 'exact'] | None = None , ttm_method : Literal['sum' , 'avg'] | None = None):
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date, lastn , qtr_method , ttm_method)

@singleton
class BalanceSheetAccess(FDataAccess):
    SINGLE_TYPE = 'balance'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'exact'
    DEFAULT_TTM_METHOD : Literal['sum' , 'avg'] = 'avg'

@singleton
class CashFlowAccess(FDataAccess):
    SINGLE_TYPE = 'cashflow'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    DEFAULT_TTM_METHOD : Literal['sum' , 'avg'] = 'sum'

@singleton
class IncomeStatementAccess(FDataAccess):
    SINGLE_TYPE = 'income'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    DEFAULT_TTM_METHOD : Literal['sum' , 'avg'] = 'sum'
  
@singleton
class FinancialDataAccess(FDataAccess):
    DATA_TYPE_LIST = ['dividend' , 'disclosure' , 'express' , 'forecast' , 'mainbz']
    DATA_TYPE_LIST_TYPE = Literal['dividend' , 'disclosure' , 'express' , 'forecast' , 'mainbz']

INDI = IndicatorDataAccess()
BS   = BalanceSheetAccess()
CF   = CashFlowAccess()
IS   = IncomeStatementAccess()
FINA = FinancialDataAccess()