import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.singleton import singleton_threadsafe

def phigh(date , n_months : int , lag_months : int = 0):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    high = TSData.TRADE.get_quotes(start_date,end_date,'high',pivot=True).max()
    cp   = TSData.TRADE.get_quotes(end_date,end_date,'close',pivot=True).iloc[-1]
    mom  = cp / high - 1
    return mom

@singleton_threadsafe
class mom_phigh1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 1)

@singleton_threadsafe
class mom_phigh2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '2个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 2)
    
@singleton_threadsafe
class mom_phigh3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '3个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 3)
    
@singleton_threadsafe
class mom_phigh6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '6个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 6)
    
@singleton_threadsafe
class mom_phigh12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 12)