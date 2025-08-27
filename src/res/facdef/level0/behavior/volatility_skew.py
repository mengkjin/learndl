import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


def skewness_volwei(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    vol = DATAVENDOR.TRADE.get_quotes(start_date,end_date,'volume',pivot=True) + 1e-6
    ret = DATAVENDOR.TRADE.get_returns(start_date,end_date,'close',pivot=True)

    wgt = vol / vol.mean()
    ret -= (ret * wgt).mean()

    wgt = vol.pow(1/3) / vol.pow(1/3).mean()
    return (ret * wgt).skew()

class price_weiskew1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '1个月成交量加权偏度'
    def calc_factor(self , date : int):
        return skewness_volwei(date , 1)
    
class price_weiskew2m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '2个月成交量加权偏度'
    def calc_factor(self , date : int):
        return skewness_volwei(date , 2)
    
class price_weiskew3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '3个月成交量加权偏度'
    def calc_factor(self , date : int):
        return skewness_volwei(date , 3)
    
class price_weiskew6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '6个月成交量加权偏度'
    def calc_factor(self , date : int):
        return skewness_volwei(date , 6)
    
class price_weiskew12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'volatility'
    description = '12个月成交量加权偏度'
    def calc_factor(self , date : int):
        return skewness_volwei(date , 12)
    