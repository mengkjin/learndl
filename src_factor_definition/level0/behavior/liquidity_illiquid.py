import numpy as np
import pandas as pd

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def amihud(date , n_months : int , lag_months : int = 0):
    '''
    Amihud illiquidity factor
    '''
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    ret = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot=True)
    vol = DATAVENDOR.TRADE.get_volumes(start_date , end_date , volume_type='volume' , pivot=True)
    amihud = (DATAVENDOR.TRADE.mask_min_finite(ret).abs() / vol).mean() * (10 ** 6)
    return amihud

def mif(date , n_months : int , lag_months : int = 0):
    '''
    market impact factor
    '''
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    vwap = DATAVENDOR.TRADE.get_quotes(start_date , end_date , 'vwap' , pivot=True)
    cp   = DATAVENDOR.TRADE.get_quotes(start_date , end_date , 'close' , pivot=True)
    vol = DATAVENDOR.TRADE.get_volumes(start_date , end_date , volume_type='amount' , pivot=True)
    mif = (DATAVENDOR.TRADE.mask_min_finite(vwap / cp - 1) / vol).sum() * (10 ** 6)
    return mif

class illiq_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月Amihud非流动性因子'
    
    def calc_factor(self, date: int):
        return amihud(date , 1)
    
class illiq_2m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '2个月Amihud非流动性因子'
    
    def calc_factor(self, date: int):
        return amihud(date , 2)
    
class illiq_3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '3个月Amihud非流动性因子'
    
    def calc_factor(self, date: int):
        return amihud(date , 3)
    
class illiq_6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '6个月Amihud非流动性因子'
    
    def calc_factor(self, date: int):
        return amihud(date , 6)
    
class illiq_12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '12个月Amihud非流动性因子'
    
    def calc_factor(self, date: int):
        return amihud(date , 12)
    
class mif_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月市场冲击因子'
    
    def calc_factor(self, date: int):
        return mif(date , 1)