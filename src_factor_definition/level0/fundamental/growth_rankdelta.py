import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'eps_rank_delta' , 'sales_rank_delta' , 'gp_rank_delta' , 'npro_rank_delta' , 
    'dedt_rank_delta' , 'tax_rank_delta' , 'roe_rank_delta' , 
    'gp_margin_rank_delta' , 'oper_margin_rank_delta' , 'net_margin_rank_delta' ,
    'ta_rank_delta' , 'equ_rank_delta' , 'liab_rank_delta' , 'cfo_rank_delta' ,
]

def get_indrank_delta(expression : str , date : int):
    '''4 quarter delta of industry ranking'''
    df = DATAVENDOR.get_fin_hist(expression , date , 5 , new_name = 'value')
    df = DATAVENDOR.INFO.add_indus(df , date , 'unknown')

    df['value'] = df.groupby(['end_date' , 'indus'])['value'].rank(pct=True)
    df = df.drop(columns = ['indus'])
    return (df - df.groupby('secid').shift(4)).dropna().groupby('secid').last().iloc[:,0]

class eps_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM每股收益行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('eps@ttm' , date)

class sales_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('sales@ttm' , date)
    
class gp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('gp@ttm' , date)
    
class npro_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('npro@ttm' , date)
    
class dedt_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM扣非归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('dedt@ttm' , date)
    
class tax_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('tax@ttm' , date)

class roe_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净资产收益率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('npro@ttm / equ@ttm' , date)
    
class gp_margin_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('gp@ttm / sales@ttm' , date)

class oper_margin_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM营业利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('oper_np@ttm / sales@ttm' , date)

class net_margin_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净利率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('npro@ttm / sales@ttm' , date)
    
class ta_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总资产行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('ta@qtr' , date)   
    
class equ_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净资产行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('equ@qtr' , date)
    
class liab_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总负债行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('liab@qtr' , date)
    
class cfo_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '经营活动现金流行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_indrank_delta('ncfo@qtr' , date)
