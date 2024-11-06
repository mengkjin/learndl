import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.singleton import singleton_threadsafe

def mdr(date , n_months : int , lag_months : int = 0):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    returns = TSData.TRADE.get_returns(start_date , end_date)
    return returns.max()

@singleton_threadsafe
class mom_mdr1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 1)

@singleton_threadsafe
class mom_mdr2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '2个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 2)
    
@singleton_threadsafe
class mom_mdr3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '3个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 3)
    
@singleton_threadsafe
class mom_mdr6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '6个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 6)
    
@singleton_threadsafe
class mom_mdr12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 12)