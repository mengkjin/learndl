import numpy as np
import pandas as pd

from typing import Literal
from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def mom_classic(date , n_months : int , lag_months : int = 0 , return_type : Literal['close' , 'overnight' , 'intraday'] = 'close'):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , return_type = return_type , mask = True)
    mom = (1 + rets).prod() - 1
    return mom

class mom_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 1)
    
class mom_2m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '2个月动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 2)
    
class mom_3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '3个月动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 3)

class mom_6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '6个月动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 6)
    
class mom_12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 12)

class mom_12m_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月动量因子(间隔1月)'

    def calc_factor(self , date : int):
        return mom_classic(date , 12 , 1)
    
class mom_1m_intraday(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月日内动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 1 , return_type='intraday')
    
class mom_1m_overnight(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月日间动量因子'

    def calc_factor(self , date : int):
        return mom_classic(date , 1 , return_type='overnight')

class mom_new1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月日内-日间合成因子'

    def calc_factor(self , date : int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm' , 0)
        rets_intraday = DATAVENDOR.TRADE.get_returns(start_date , end_date , return_type = 'intraday' , mask = True)
        rets_overnight = DATAVENDOR.TRADE.get_returns(start_date , end_date , return_type = 'overnight' , mask = True)

        mom = rets_intraday.rank(axis = 1 , pct = True) - rets_overnight.rank(axis = 1 , pct = True)
        return mom.sum()