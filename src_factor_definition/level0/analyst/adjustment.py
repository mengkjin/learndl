import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'rec_npro_12m' , 'rec_npro_3m' , 'rec_npro_6m' ,
    'up_npro_dt12m' , 'up_npro_dt6m' ,
    'uppct_npro_12m' , 'uppct_npro_3m' , 'uppct_npro_6m' ,
    'uppct_npro_dt12m' , 'uppct_npro_dt6m'
]

class rec_npro_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '12个月盈利预测调整因子'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'rec_npro_12m')
        return df
    
class rec_npro_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '3个月盈利预测调整因子'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'rec_npro_3m')
        return df
    
class rec_npro_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '6个月盈利预测调整因子'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'rec_npro_6m')
        return df

class up_npro_dt12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '12个月分析师盈利上修数量--公告日'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'up_npro_dt12m')
        return df  

class up_npro_dt6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '6个月分析师盈利上修数量--公告日'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'up_npro_dt6m')
        return df  

class uppct_npro_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '12个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'uppct_npro_12m')
        return df
    
class uppct_npro_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '3个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'uppct_npro_3m')
        return df

class uppct_npro_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '6个月分析师盈利上修占比'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'uppct_npro_6m')
        return df

class uppct_npro_dt12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '12个月分析师盈利上修占比--公告日'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'uppct_npro_dt12m')
        return df
    
class uppct_npro_dt6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'adjustment'
    description = '6个月分析师盈利上修占比--公告日'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_adjustment' , date , 'uppct_npro_dt6m')
        return df
