import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import DATAVENDOR

def exret_std(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    exrets = DATAVENDOR.RISK.get_exret(start_date , end_date , pivot=True)
    return exrets.std() * np.sqrt(252)

class exret_std1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '1个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 1)
    
class exret_std2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '2个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 2)
 
class exret_std3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '3个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 3)
    
class exret_std6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '6个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 6)
    
class exret_std12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '12个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 12)