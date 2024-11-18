import numpy as np
import pandas as pd

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def phigh(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    high = DATAVENDOR.TRADE.get_quotes(start_date,end_date,'high',pivot=True).max()
    cp   = DATAVENDOR.TRADE.get_quotes(end_date,end_date,'close',pivot=True).iloc[-1]
    mom  = cp / high - 1
    return mom

class mom_phigh1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 1)