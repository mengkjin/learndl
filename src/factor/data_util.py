import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import Literal

from src.data import DataBlock
from src.data.core import GetData
from src.data.process import BlockLoader
from src.environ import RISK_STYLE

class DataVendor:
    def __init__(self):
        self.start_dt = 99991231
        self.end_dt   = -1
        self.max_date   = GetData.data_dates('trade' , 'day')[-1]
        self.trade_date = GetData.trade_dates()
        self.all_stocks = GetData.stocks().sort_values('secid')

    def td_within(self , start_dt : int = -1 , end_dt : int = 99991231 , step : int = 1):
        return self.trade_date[(self.trade_date >= start_dt) & (self.trade_date <= end_dt)][::step]

    def td_offset(self , date , offset : int = 0):
        if np.isscalar(date):
            if date not in self.trade_date: date = self.trade_date[self.trade_date <= date][-1]
            d = self.trade_date[np.where(self.trade_date == date)[0][0] + offset]
            return int(d)
        else:
            return np.array([self.td_offset(d , offset) for d in date])

    def random_factor(self):
        date  = self.td_within(20240101,20240531,5)
        secid = self.all_stocks.secid
        factor_val = DataBlock(np.random.randn(len(secid),len(date),1,2),secid,date,['factor1','factor2'])
        return factor_val

    def get_returns(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt)
        if (not hasattr(self , 'day_ret')) or (not np.isin(td_within , self.day_ret.date).all()):
            self.day_ret  = GetData.daily_returns(start_dt , end_dt)

    def get_risk_exp(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt)
        if (not hasattr(self , 'risk_exp')) or (not np.isin(td_within , self.risk_exp.date).all()):
            with GetData.Silence():
                self.risk_exp = BlockLoader('models', 'risk_exp').load_block(start_dt , end_dt)

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 , 
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        assert lag > 0 , f'lag must be positive : {lag}'
        date_min  = int(self.td_offset(date.min() , -10))
        date_max  = int(self.td_offset(int(date.max()) , nday + lag + 10))
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
    
DATAVENDOR = DataVendor()