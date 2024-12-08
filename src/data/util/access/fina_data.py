import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass , field
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
    DEFAULT_QOQ_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'qtr'
    DEFAULT_YOY_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'ttm'

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

    def _fin_hist_data_transform(self , df : pd.DataFrame , val : str , benchmark_df : pd.DataFrame | None = None , 
                                 lastn = 1 , pivot = False , ffill = False):
        if benchmark_df is not None:
            df = df.reindex(benchmark_df.index).where(~benchmark_df.isna() , np.nan)
        if ffill: df = df.groupby('secid').ffill()
        df = df.groupby('secid').tail(lastn).replace([np.inf , -np.inf] , np.nan)
        if pivot: df = df.pivot_table(val , 'end_date' , 'secid').sort_index()
        return df

    def _get_data_acc_hist(self , data_type : str , val : str , date : int , lastn = 1 , 
                           pivot = False , ffill = False , year_only = False):
        assert data_type in ['income' , 'cashflow' , 'balance' , 'indicator'] , \
            f'invalid data_type: {data_type} , must be one of ' + str(['income' , 'cashflow' , 'balance' , 'indicator'])
        q_ends = self.qtr_ends(date , lastn + 4 , year_only = year_only)

        field = ['secid' , 'ann_date' , 'update_flag' , val]
        
        df_acc = self.gets(q_ends , data_type , field , rename_date_key = 'end_date').\
            reset_index(drop = False).dropna(subset = ['ann_date' , 'end_date'])
        # df_acc = pd.concat([self.get(qe , data_type , field) for qe in q_ends]).dropna(subset = ['ann_date' , 'end_date'])
        df_acc['ann_date'] = df_acc['ann_date'].astype(int)
        df_acc['end_date'] = df_acc['end_date'].astype(int)
        df_acc = df_acc[(df_acc['ann_date'] <= date) & df_acc['end_date'].isin(q_ends) & (df_acc['secid'] >= 0)]
        df_acc = df_acc.sort_values('update_flag').drop_duplicates(['secid' , 'end_date'] , keep = 'last')\
            [['secid','end_date',val]].set_index(['secid' , 'end_date']).sort_index()
        return self._fin_hist_data_transform(df_acc , val , None , lastn , pivot , ffill)
    
    def _get_data_qtr_hist(self , data_type : str , val : str , date : int , lastn = 1 , 
                           pivot = False , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD
        if qtr_method == 'exact':
            return self._get_data_acc_hist(data_type , val , date , lastn , pivot , ffill)

        df_acc = self._get_data_acc_hist(data_type , val , date , lastn + 4 , pivot = False)
        secid = df_acc.index.get_level_values('secid').unique()
        q_end = df_acc.index.get_level_values('end_date').unique()
        ystarts = np.unique(q_end // 10000) * 10000
        df_ystarts = pd.DataFrame({val:0} , index = pd.MultiIndex.from_product([secid , ystarts] , names=['secid' , 'end_date']))
        df_qtr = pd.concat([df_acc , df_ystarts]).sort_index().groupby('secid').ffill().fillna(0)
        
        return self._fin_hist_data_transform(df_qtr , val , df_acc , lastn , pivot , ffill)

    def _get_data_ttm_hist(self , data_type : str , val : str , date : int , lastn = 1 , 
                           pivot = False , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD 
        df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 8 , pivot = False , ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_ttm = df_qtr.reindex(full_index).fillna(0)
        grp = df_ttm.groupby('secid')
        if qtr_method == 'diff': df_ttm = grp.rolling(4).sum()
        elif qtr_method == 'exact': df_ttm = grp.rolling(5).mean()
        df_ttm = df_ttm.reset_index(level=0, drop=True)
        return self._fin_hist_data_transform(df_ttm , val , df_qtr , lastn , pivot , ffill)
    
    def _get_data_qoq_hist(self , data_type : str , val : str , date : int , lastn = 1 , 
                           pivot = False , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None , 
                           qoq_method : Literal['ttm' , 'acc' , 'qtr'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD 
        if qoq_method is None: qoq_method = self.DEFAULT_QOQ_METHOD
        if qoq_method == 'qtr':
            df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 2 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        elif qoq_method == 'ttm':
            df_qtr = self._get_data_ttm_hist(data_type , val , date , lastn + 2 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_qoq = df_qtr.reindex(full_index)
        df_qoq_base = df_qoq.groupby('secid').shift(1)
        df_qoq = (df_qoq - df_qoq_base) / df_qoq_base.abs()
        return self._fin_hist_data_transform(df_qoq , val , df_qtr , lastn , pivot , ffill)
    
    def _get_data_yoy_hist(self , data_type : str , val : str , date : int , lastn = 1 , 
                           pivot = False , ffill = False , 
                           qtr_method : Literal['diff' , 'exact'] | None = None ,
                           yoy_method : Literal['ttm' , 'acc' , 'qtr'] | None = None):
        if qtr_method is None: qtr_method = self.DEFAULT_QTR_METHOD 
        if yoy_method is None: yoy_method = self.DEFAULT_YOY_METHOD
        if yoy_method == 'ttm':
            df_qtr = self._get_data_ttm_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        elif yoy_method == 'acc':
            df_qtr = self._get_data_acc_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False)
        elif yoy_method == 'qtr':
            df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_yoy = df_qtr.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(4)
        df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()
        return self._fin_hist_data_transform(df_yoy , val , df_qtr , lastn , pivot , ffill)
    
    def _fin_latest_data_transform(self , df : pd.DataFrame , val : str):
        return df.dropna().groupby('secid').last()[val]

    def _get_data_acc_latest(self , data_type : str , val : str , date : int):
        df_acc = self._get_data_acc_hist(data_type , val , date , 3)
        return self._fin_latest_data_transform(df_acc , val)
    
    def _get_data_qtr_latest(self , data_type : str , val : str , date : int , 
                             qtr_method : Literal['diff' , 'exact'] | None = None):
        df_qtr = self._get_data_qtr_hist(data_type , val , date , 3 , ffill = False , qtr_method = qtr_method)
        return self._fin_latest_data_transform(df_qtr , val)
    
    def _get_data_ttm_latest(self , data_type : str , val : str , date : int , 
                             qtr_method : Literal['diff' , 'exact'] | None = None):
        df_ttm = self._get_data_ttm_hist(data_type , val , date , 3 , ffill = False , qtr_method = qtr_method)
        return self._fin_latest_data_transform(df_ttm , val)
    
    def _get_data_qoq_latest(self , data_type : str , val : str , date : int , 
                             qtr_method : Literal['diff' , 'exact'] | None = None ,
                             qoq_method : Literal['qtr' , 'ttm'] | None = None):
        df_qoq = self._get_data_qoq_hist(data_type , val , date , 3 , 
                                         qtr_method = qtr_method , qoq_method = qoq_method)
        return self._fin_latest_data_transform(df_qoq , val)
    
    def _get_data_yoy_latest(self , data_type : str , val : str , date : int , 
                             qtr_method : Literal['diff' , 'exact'] | None = None ,
                             yoy_method : Literal['ttm' , 'acc'] | None = None):
        df_yoy = self._get_data_yoy_hist(data_type , val , date , 3 , qtr_method = qtr_method , yoy_method = yoy_method)
        return self._fin_latest_data_transform(df_yoy , val)
    
    @property
    def fields(self):
        assert self.SINGLE_TYPE is not None , 'SINGLE_TYPE must be set'
        if self.collections[self.SINGLE_TYPE]: 
            return self.collections[self.SINGLE_TYPE].columns()
        else:
            return self.get(20231231 , self.SINGLE_TYPE).columns.values

    def acc(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , year_only = False):
        return self._get_data_acc_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , year_only)
    
    def qtr(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False):
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , self.DEFAULT_QTR_METHOD)

    def ttm(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False):
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , self.DEFAULT_QTR_METHOD)
    
    def qoq(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , self.DEFAULT_QTR_METHOD , qoq_method)
    
    def yoy(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , self.DEFAULT_QTR_METHOD , yoy_method)

    def acc_latest(self, val: str, date: int):
        return self._get_data_acc_latest(self.SINGLE_TYPE , val, date)
    
    def qtr_latest(self, val: str, date: int):
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date, self.DEFAULT_QTR_METHOD)
    
    def ttm_latest(self, val: str, date: int):
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date, self.DEFAULT_QTR_METHOD)
    
    def qoq_latest(self, val: str, date: int , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_latest(self.SINGLE_TYPE , val, date, self.DEFAULT_QTR_METHOD , qoq_method)
    
    def yoy_latest(self, val: str, date: int , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_latest(self.SINGLE_TYPE , val, date, self.DEFAULT_QTR_METHOD , yoy_method)
    
@singleton
class IndicatorDataAccess(FDataAccess):
    SINGLE_TYPE = 'indicator'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'

    def qtr(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method)

    def ttm(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method)
    
    def qoq(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , qoq_method)
    
    def yoy(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , yoy_method)
    
    def qtr_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date , qtr_method)
    
    def ttm_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date , qtr_method)
    
    def qoq_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_latest(self.SINGLE_TYPE , val, date , qtr_method , qoq_method)
    
    def yoy_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_latest(self.SINGLE_TYPE , val, date , qtr_method , yoy_method)

@singleton
class BalanceSheetAccess(FDataAccess):
    SINGLE_TYPE = 'balance'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'exact'
    DEFAULT_YOY_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'acc'
    DEFAULT_QOQ_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'acc'

@singleton
class CashFlowAccess(FDataAccess):
    SINGLE_TYPE = 'cashflow'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'

@singleton
class IncomeStatementAccess(FDataAccess):
    SINGLE_TYPE = 'income'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
  
@singleton
class FinancialDataAccess(FDataAccess):
    DATA_TYPE_LIST = ['dividend' , 'disclosure' , 'express' , 'forecast' , 'mainbz']
    DATA_TYPE_LIST_TYPE = Literal['dividend' , 'disclosure' , 'express' , 'forecast' , 'mainbz']

INDI = IndicatorDataAccess()
BS   = BalanceSheetAccess()
CF   = CashFlowAccess()
IS   = IncomeStatementAccess()
FINA = FinancialDataAccess()

@dataclass(slots=True)
class FinData:
    statement : Literal['is' , 'cf' , 'indi' , 'bs'] | str
    val : str
    fin_type : Literal['ttm' , 'qtr' , 'acc' , 'yoy' , 'qoq'] | str
    kwargs : dict[str,Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls , numerator : str , **kwargs):
        numerator , kwargs = cls.reserved_numerator(numerator , **kwargs)
        components = numerator.split('@')
        assert len(components) == 3 , 'invalid numerator: ' + numerator
        assert components[0] in ['is' , 'cf' , 'indi' , 'bs'] , components[0]
        assert components[2] in ['ttm' , 'qtr' , 'acc' , 'yoy' , 'qoq'] , components[2]
        return cls(components[0] , components[1] , components[2] , kwargs)
    
    @classmethod
    def reserved_numerator(cls , numerator : str , **kwargs):
        first_signal = numerator.split('@')[0]
        reserved_signals = {
            
            'ta' : 'bs@total_assets' ,
            'equ' : 'bs@total_hldr_eqy_exc_min_int' ,

            'sales' : 'is@revenue' ,
            'oper_np' : 'is@operate_profit' ,
            'total_np' : 'is@total_profit' ,
            'npro' : 'is@n_income_attr_p' ,
            'ebit' : 'is@ebit' ,
            'ebitda' : 'is@ebitda' ,
            
            'nocf' : 'cf@n_cashflow_act' ,

            'dedt' : 'indi@profit_dedt' ,
            'eps' : 'indi@eps' ,
            'gp' : 'indi@gross_margin' ,
        }
        if first_signal not in reserved_signals:
            ...
        else:
            numerator = numerator.replace(f'{first_signal}@' , f'{reserved_signals[first_signal]}@')
            if first_signal in ['gp' , 'dedt']:
                kwargs = {'qtr_method' : 'diff'} | kwargs
        return numerator , kwargs
    
    def get_source(self):
        if self.statement == 'is': return IS
        elif self.statement == 'cf': return CF
        elif self.statement == 'indi': return INDI
        elif self.statement == 'bs': return BS
        else: raise ValueError(f'invalid statement: {self.statement}')
    
    def get_latest(self):
        src = self.get_source()
        func = getattr(src , f'{self.fin_type}_latest')
        return func(self.val , **self.kwargs)
    
    def get_hist(self):
        src = self.get_source()
        func = getattr(src , f'{self.fin_type}')
        return func(self.val , **self.kwargs)