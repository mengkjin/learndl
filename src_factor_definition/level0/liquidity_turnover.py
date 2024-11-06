import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.singleton import singleton_threadsafe

def turnover_classic(date , n_months : int , lag_months : int = 0 , min_finite_ratio = 0.25):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    turns = TSData.TRADE.mask_min_finite(TSData.TRADE.get_turnovers(start_date , end_date , turnover_type = 'fr') , 
                                         min_finite_ratio = min_finite_ratio)
    turns = turns.mean()
    return turns

@singleton_threadsafe
class turn_1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 1)

@singleton_threadsafe
class turn_2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '2个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 2)

@singleton_threadsafe
class turn_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '3个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 3)
    
@singleton_threadsafe
class turn_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '6个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 6)
    
@singleton_threadsafe
class turn_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '12个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 12)
    
@singleton_threadsafe
class turn_unexpected(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '意外换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 1 , 0) - turnover_classic(date , 3 , 1)