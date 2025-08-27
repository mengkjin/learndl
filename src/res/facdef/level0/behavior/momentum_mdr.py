import numpy as np
import pandas as pd

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


def mdr(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    returns = DATAVENDOR.TRADE.get_returns(start_date , end_date)
    return returns.max()

class mom_mdr1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 1)

class mom_mdr2m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '2个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 2)
    
class mom_mdr3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '3个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 3)
    
class mom_mdr6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '6个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 6)
    
class mom_mdr12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 12)