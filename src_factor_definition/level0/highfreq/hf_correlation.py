import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'beta_intra' , 'corr_intra' , 'pvcorrstd_intra' , 'rar_intra' , 'vol_vwapcorr' , 'volar_intra'
]

class beta_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '日内beta'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'beta_intra')
        return df
    
class corr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '日内量价相关性'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'corr_intra')
        return df
    
class pvcorrstd_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '价量相关性波动性'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'pvcorrstd_intra')
        return df
    
class rar_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '收益率自相关系数'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'rar_intra')
        return df
    
class vol_vwapcorr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '波动价相关性'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'vol_vwapcorr')
        return df
    
class volar_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '成交量自相关系数'

    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('high_frequency' , date , 'volar_intra')
        return df
