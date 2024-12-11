import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'err_intra' , 'maxdd_intra' , 'skew_intra1min' , 'skew_intra5min' , 'vardown_intra1min' ,
    'vardown_intra5min' , 'vol_stdwei' , 'vov_intra'
]

class err_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内极端收益'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'err_intra')
        return df
    
class maxdd_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内最大回撤'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'maxdd_intra')
        return df
    
class skew_intra1min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '高频偏度'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'skew_intra1min')
        return df
    
class skew_intra5min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '高频偏度'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'skew_intra5min')
        return df
    
class vardown_intra1min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '下行波动占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vardown_intra1min')
        return df
    
class vardown_intra5min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '下行波动占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vardown_intra5min')
        return df
    
class vol_stdwei(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '波动加权成交占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_stdwei')
        return df

class vov_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '日内波动率'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vov_intra')
        return df