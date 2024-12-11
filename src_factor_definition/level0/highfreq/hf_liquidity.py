import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'smart_money' , 'tcv_intra' , 'turn_utd' , 'varr_intra' , 'vol_end15min' , 'vol_st5min' ,
    'volpct_phigh' , 'volpct_plow' , 'volvr_intra'
]

class smart_money(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '聪明钱因子'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'smart_money')
        return df
    
class tcv_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '一致买入因子'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'tcv_intra')
        return df

class turn_utd(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '换手率分布'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'turn_utd')
        return df
    
class varr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '变异数比率'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'varr_intra')
        return df
    
class vol_end15min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '尾盘成交占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_end15min')
        return df
    
class vol_st5min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '开盘成交占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_st5min')
        return df
    
class volpct_phigh(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '价格排序成交占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'volpct_phigh')
        return df
    
class volpct_plow(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '价格排序成交占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'volpct_plow')
        return df
    
class volvr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_liquidity'
    description = '成交量占比'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'volvr_intra')
        return df
