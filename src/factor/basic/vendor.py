import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import Literal

from src.data import DataBlock , GetData , BlockLoader
from src.environ import RISK_STYLE , RISK_INDUS

class DataVendor:
    def __init__(self):
        self.start_dt = 99991231
        self.end_dt   = -1
        self.max_date   = GetData.data_dates('trade' , 'day')[-1]
        self.trade_date = GetData.trade_dates()
        self.all_stocks = GetData.stocks().sort_values('secid')
        self.st_stocks  = GetData.st_stocks()

    @property
    def secid(self): return self.all_stocks.secid.unique()

    def td_within(self , start_dt : int = -1 , end_dt : int = 99991231 , step : int = 1):
        return self.trade_date[(self.trade_date >= start_dt) & (self.trade_date <= end_dt)][::step]

    def td_offset(self , date , offset : int = 0):
        if np.isscalar(date):
            assert isinstance(date , int) , date
            if date not in self.trade_date: date = self.trade_date[self.trade_date <= date][-1]
            if offset: date = self.trade_date[np.where(self.trade_date == date)[0][0] + offset]
            return int(date)
        else:
            return np.array([self.td_offset(d , offset) for d in date])
    
    def latest_td(self , date : int): return self.td_offset(date)

    def random_factor(self , start_dt = 20240101 , end_dt = 20240531 , step = 5 , nfactor = 2):
        date  = self.td_within(start_dt , end_dt , step)
        secid = self.secid
        factor_val = DataBlock(np.random.randn(len(secid),len(date),1,nfactor),
                               secid,date,[f'factor{i+1}' for i in range(nfactor)])
        return factor_val

    def get_returns(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt)
        if (not hasattr(self , 'day_ret')) or (not np.isin(td_within , self.day_ret.date).all()):
            self.day_ret  = GetData.daily_returns(start_dt , end_dt)

    def update_dates(self , data_key : str , dates : np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        if hasattr(self , data_key):
            exist_dates = getattr(self , data_key).date
            if len(exist_dates):
                early_dates = dates[dates < exist_dates.min()]
                late_dates  = dates[dates > exist_dates.min()]
            else:
                early_dates = dates
                late_dates  = np.array([])
        else:
            early_dates = dates
            late_dates  = np.array([])
        return early_dates , late_dates

    def get_named_data_block(self , start_dt : int , end_dt : int , db_src , db_key , data_key):
        td_within = self.td_within(start_dt , end_dt)
        if len(td_within) == 0: return
        with GetData.Silence:
            early_dates , late_dates = self.update_dates(data_key , td_within)
            datas : list[DataBlock] = []
            if len(early_dates): datas.append(BlockLoader(db_src , db_key).load_block(early_dates.min() , early_dates.max()))
            if len(late_dates):  datas.append(BlockLoader(db_src , db_key).load_block(late_dates.min() , late_dates.max()))
            if hasattr(self , data_key): datas.append(getattr(self , data_key))
        return DataBlock.merge(datas)

    def get_daily_quote(self , start_dt : int , end_dt : int):
        db_src , db_key = 'trade' , 'day'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'daily_quote')
        if isinstance(data , DataBlock):
            self.daily_quote = data
        return

    def get_risk_exp(self , start_dt : int , end_dt : int):
        db_src , db_key = 'models' , 'risk_exp'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'risk_exp')
        if isinstance(data , DataBlock):
            self.risk_exp = data
        return

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 , 
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        assert lag > 0 , f'lag must be positive : {lag}'
        date_min = int(self.td_offset(date.min() , -10))
        date_max = int(self.td_offset(int(date.max()) , nday + lag + 10))
        self.get_returns(date_min , date_max)
        full_date = self.td_within(date_min , date_max)

        block = self.day_ret.align(secid , full_date , [ret_type] , inplace=False).as_tensor()
        block.values = F.pad(block.values[:,lag:] , (0,0,0,0,0,lag) , value = torch.nan)

        new_value = block.values.unfold(1 , nday , 1).exp().prod(dim = -1) - 1
        feature   = ['ret']

        new_block = DataBlock(new_value , secid , full_date[:new_value.shape[1]] , feature).align_date(date)
        return new_block
    
    def risk_style_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , RISK_STYLE , inplace=False).as_tensor()
        return block
    
    def risk_industry_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , RISK_INDUS , inplace=False).as_tensor()
        return block
    
    def get_ffmv(self , secid : np.ndarray , d : int):
        if d not in self.trade_date: return None
        self.get_risk_exp(d , d)
        value = self.risk_exp.loc(secid = secid , date = d , feature = 'weight').flatten()
        return value
    
    def get_cp(self , secid : np.ndarray , d : int):
        if d not in self.trade_date: return None
        self.get_daily_quote(d , d)
        value = self.daily_quote.loc(secid = secid , date = d , feature = 'close').flatten()
        return value
    
DATAVENDOR = DataVendor()