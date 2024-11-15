import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import DATAVENDOR


def ret_std_classic(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    returns = DATAVENDOR.TRADE.get_returns(start_date , end_date)
    return returns.std() * np.sqrt(252)

class ret_std1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '1个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 1)
    
class ret_std2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '2个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 2)
    
class ret_std3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '3个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 3)
    
class ret_std6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '6个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 6)
    
class ret_std12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '12个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 12)