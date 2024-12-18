import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'ta_gro5y' , 'equ_gro5y' , 'liab_gro5y' , 'sales_gro5y' , 'npro_gro5y' , 'cfo_gro5y'
]

def get_compound_growth(expression: str , date: int , n_year : int = 5 , **kwargs):
    '''cannot deal with < -100% growth compounding, use simple instead'''
    df = DATAVENDOR.get_fin_hist(expression , date , 4*n_year + 1 , pivot = False).iloc[:,0].reset_index('end_date',drop=False)
    df = pd.concat([df.groupby('secid').first() , df.groupby('secid').last()], axis=0)
    val = df.columns[-1]
    
    df['qtrs'] = (df['end_date'] // 10000) * 4 + df['end_date'] % 10000 // 300 
    df = df.set_index('end_date',append=True).sort_index()

    # df = (df.groupby('secid')[val].pct_change() + 1) ** (4 / df.groupby('secid')['qtrs'].diff()) - 1
    df = df.groupby('secid')[val].pct_change() * 4 / df.groupby('secid')['qtrs'].diff()
    return df.groupby('secid').last()

class ta_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-总资产'
    
    def calc_factor(self, date: int):
        return get_compound_growth('ta@qtr' , date)
    
class equ_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-净资产'
    
    def calc_factor(self, date: int):
        return get_compound_growth('equ@qtr' , date)
    
class liab_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-总负债'
    
    def calc_factor(self, date: int):
        return get_compound_growth('liab@qtr' , date)

class sales_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-营业收入'
    
    def calc_factor(self, date: int):
        return get_compound_growth('sales@ttm' , date)

class npro_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-归母净利润'
    
    def calc_factor(self, date: int):
        return get_compound_growth('npro@ttm' , date)
    
class cfo_gro5y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-经营活动现金流'
    
    def calc_factor(self, date: int):
        return get_compound_growth('ncfo@ttm' , date)
