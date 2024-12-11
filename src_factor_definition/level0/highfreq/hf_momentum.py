import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'apm_orig' , 'conf_persit' , 'hphigh_intra' , 'mom_uret' , 'pvi_intra' , 'trendratio_intra' ,
    'vol_highret' , 'vol_highvwap' , 'vol_lowvwap'
]

class apm_orig(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = 'APM原始值'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'apm_orig')
        return df
    
class conf_persit(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '过度自信因子'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'conf_persit')
        return df
    
class hphigh_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '日内高点位置'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'hphigh_intra')
        return df
    
class mom_uret(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '动量因子'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'mom_uret')
        return df
    
class pvi_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '量升累计收益'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'pvi_intra')
        return df
    
class trendratio_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '日内价格变化路径'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'trendratio_intra')
        return df
    
class vol_highret(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '高波动收益率'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_highret')
        return df
    
class vol_highvwap(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '高成交量交易成本'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_highvwap')
        return df
    
class vol_lowvwap(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_momentum'
    description = '低成交量交易成本'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_lowvwap')
        return df