import torch
import numpy as np
import pandas as pd
from typing import Any , Literal

from .calendar import TradeCalendar , TradeDate
from ....basic import PATH
from ....func.time import today
from ....func.singleton import singleton_threadsafe

@singleton_threadsafe
class TradeDataAccess:
    MAX_LEN = 1000

    def __init__(self) -> None:
        self.day_val : dict[int , pd.DataFrame] = {}
        self.day_trd : dict[int , pd.DataFrame] = {}

    def load(self , dates):
        self.load_val(dates)
        self.load_trd(dates)
    
    def load_val(self, dates , drop_old = True):
        if drop_old and (len(self.day_val) + len(dates) > self.MAX_LEN):  self.day_val = {}
        for d in dates: self.day_val[d] = PATH.load_target_file('trade_ts' , 'day_val' , d)

    def load_trd(self, dates , drop_old = True):
        if drop_old and (len(self.day_trd) + len(dates) > self.MAX_LEN):  self.day_trd = {}
        for d in dates: self.day_trd[d] = PATH.load_target_file('trade_ts' , 'day' , d)

    def get_val(self , date , cols = None):
        if date not in self.day_val:  self.load_val([date] , drop_old=False)
        df = self.day_val[date]
        if df is not None: df = df.assign(date = date)
        if cols is not None: df = df.loc[:,cols]
        return df
    
    def get_trd(self , date , cols = None):
        if date not in self.day_trd:  self.load_trd([date] , drop_old=False)
        df = self.day_trd[date]
        if df is not None: df = df.assign(date = date)
        if cols is not None: df = df.loc[:,cols]
        return df
    
    def get_rets(self , start_dt : int | TradeDate , end_dt : int | TradeDate):
        dates = CALENDAR.td_within(int(start_dt) , int(end_dt))
        self.load(dates)
        df_list = [self.get_trd(d , ['secid','pctchange']).assign(date = d) for d in dates]
        df = pd.concat(df_list).loc[:,['date','secid','pctchange']].pivot_table('pctchange','date','secid') / 100
        return df
    
@singleton_threadsafe
class ModelDataAccess:
    MAX_LEN = 1000

    def __init__(self) -> None:
        self.cne5_res : dict[int , pd.DataFrame] = {}
        self.cne5_exp : dict[int , pd.DataFrame] = {}

    def load(self , dates):
        self.load_res(dates)
        # self.load_exp(dates)

    def load_res(self, dates , drop_old = True):
        if drop_old and (len(self.cne5_res) + len(dates) > self.MAX_LEN):  self.cne5_res = {}
        for d in dates: self.cne5_res[d] = PATH.load_target_file('models' , 'tushare_cne5_res' , d)

    def load_exp(self, dates , drop_old = True):
        if drop_old and (len(self.cne5_exp) + len(dates) > self.MAX_LEN):  self.cne5_exp = {}
        for d in dates: self.cne5_exp[d] = PATH.load_target_file('models' , 'tushare_cne5_exp' , d)

    def get_res(self , date , cols = None):
        if date not in self.cne5_res:  self.load_res([date] , drop_old=False)
        df = self.cne5_res[date]
        if df is not None: df = df.assign(date = date)
        if cols is not None: df = df.loc[:,cols]
        return df
    
    def get_exp(self , date , cols = None):
        if date not in self.cne5_exp:  self.load_exp([date] , drop_old=False)
        df = self.cne5_exp[date]
        if df is not None: df = df.assign(date = date)
        if cols is not None: df = df.loc[:,cols]
        return df

@singleton_threadsafe
class FinaDataAccess:
    MAX_LEN = 40

    def __init__(self) -> None:
        self.QE = self.full_quarter_ends()
        self.fina_indi : dict[int , pd.DataFrame] = {}

    def load(self , dates):
        self.load_indi(dates)

    def load_indi(self, dates , drop_old = True):
        if drop_old and (len(self.fina_indi) + len(dates) > self.MAX_LEN):  self.fina_indi = {}
        for d in dates: 
            df = PATH.load_target_file('financial_ts' , 'indicator' , d)
            df['ann_date'] = df['ann_date'].fillna(99991231).astype(int)
            df['end_date'] = df['end_date'].fillna(-1).astype(int)
            self.fina_indi[d] = df

    def get_indi(self , date , cols = None):
        assert date in self.QE , date
        if date not in self.fina_indi:  self.load_indi([date] , drop_old=False)
        df = self.fina_indi[date]
        if cols is not None: df = df.loc[:,cols]
        return df
    
    @property
    def indi_cols(self):
        if self.fina_indi: 
            for df in self.fina_indi.values(): return df.columns.values
        else:
            self.load_indi([20231231])
            return self.fina_indi[20231231].columns.values
    
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

CALENDAR = TradeCalendar()
TRADE_DATA = TradeDataAccess()
MODEL_DATA = ModelDataAccess()
FINA_DATA = FinaDataAccess()