import numpy as np
import pandas as pd
from typing import Literal
from src.factor.classes import StockFactorCalculator
from src.data import TSData

def coefficient_variance(date , n_months : int , data_type : Literal['amount' , 'turnover'] , min_finite_ratio = 0.25):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm')
    if data_type == 'amount':
        vals = TSData.TRADE.get_volumes(start_date , end_date , volume_type = 'amount')
    elif data_type == 'turnover':
        vals = TSData.TRADE.get_turnovers(start_date , end_date)
    else:
        raise ValueError(f'data_type must be "amount" or "turnover" , but got {data_type}')
    vals = TSData.TRADE.mask_min_finite(vals , min_finite_ratio = min_finite_ratio)
    return vals.std() / vals.mean()

class amt_cv1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月成交额变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 1 , 'amount')

class amt_cv2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '2个月成交额变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 2 , 'amount')
    
class amt_cv3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '3个月成交额变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 3 , 'amount')
    
class amt_cv6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '6个月成交额变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 6 , 'amount')
    
class amt_cv12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '12个月成交额变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 12 , 'amount')

class turn_cv1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月换手率变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 1 , 'turnover')

class turn_cv2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '2个月换手率变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 2 , 'turnover')
    
class turn_cv3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '3个月换手率变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 3 , 'turnover')
    
class turn_cv6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '6个月换手率变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 6 , 'turnover')
    
class turn_cv12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '12个月换手率变异系数'
    
    def calc_factor(self, date: int):
        return coefficient_variance(date , 12 , 'turnover')