import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.singleton import singleton_threadsafe

def price_weivol(date , n_months : int , lag_months : int = 0):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    vol  = TSData.TRADE.get_quotes(start_date,end_date,'volume',pivot=True)
    cp   = TSData.TRADE.get_quotes(start_date,end_date,'close',pivot=True)
    weivol = (vol * cp).sum() / vol.sum() / cp.iloc[-1]
    return weivol

@singleton_threadsafe
class price_weivol1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 1)

@singleton_threadsafe
class price_weivol2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '2个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 2)
    
@singleton_threadsafe
class price_weivol3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '3个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 3)
    
@singleton_threadsafe
class price_weivol6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '6个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 6)
    
@singleton_threadsafe
class price_weivol12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '12个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 12)