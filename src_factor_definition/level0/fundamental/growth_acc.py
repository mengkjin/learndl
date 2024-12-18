import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'eps_acce' , 'sales_acce' , 'gp_acce' , 'npro_acce' , 'dedt_acce' , 
    'tax_acce' , 'roe_acce' , 'gp_margin_acce' , 'oper_margin_acce' , 'net_margin_acce' ,
    'eps_accv' , 'sales_accv' , 'gp_accv' , 'npro_accv' , 'dedt_accv' , 
    'tax_accv' , 'roe_accv' , 'gp_margin_accv' , 'oper_margin_accv' , 'net_margin_accv'
]

def calc_acce(data : pd.DataFrame | pd.Series):
    if isinstance(data , pd.Series): data = data.to_frame()
    delta1 = (data - data.groupby('secid').shift(4)) / data.groupby('secid').shift(4).abs()
    delta2 = (data.groupby('secid').shift(1) - data.groupby('secid').shift(5)) / data.groupby('secid').shift(5).abs()
    valid = data.groupby('secid').size() > 6
    df = (delta1 - delta2).groupby('secid').last().where(valid , np.nan).iloc[:,0]
    return df

def calc_accv(data : pd.DataFrame | pd.Series):
    if isinstance(data , pd.Series): data = data.to_frame()
    data1 = data.groupby('secid').tail(8)
    data2 = data.groupby('secid').shift(1).groupby('secid').tail(8)
    delta1 = (data1 - data1.groupby('secid').shift(4)) / data1.groupby('secid').std()
    delta2 = (data2 - data2.groupby('secid').shift(4)) / data2.groupby('secid').std()
    valid = data.groupby('secid').size() > 6
    df = (delta1 - delta2).groupby('secid').last().where(valid , np.nan).iloc[:,0]
    return df

def get_acce(expression : str , date : int):
    data = DATAVENDOR.get_fin_hist(expression , date , 10)
    return calc_acce(data)

def get_accv(expression : str , date : int):
    data = DATAVENDOR.get_fin_hist(expression , date , 10)
    return calc_accv(data)

class eps_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM每股收益行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('eps@ttm' , date)

class sales_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('sales@ttm' , date)
    
class gp_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('gp@ttm' , date)
    
class npro_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('npro@ttm' , date)
    
class dedt_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM扣非归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('dedt@ttm' , date)
    
class tax_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('tax@ttm' , date)

class roe_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净资产收益率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('npro@ttm / equ@ttm' , date)
    
class gp_margin_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('gp@ttm / sales@ttm' , date)

class oper_margin_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM营业利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('oper_np@ttm / sales@ttm' , date)

class net_margin_acce(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净利率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_acce('npro@ttm / sales@ttm' , date)
    
class eps_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM每股收益行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('eps@ttm' , date)

class sales_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('sales@ttm' , date)
    
class gp_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('gp@ttm' , date)
    
class npro_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('npro@ttm' , date)
    
class dedt_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM扣非归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('dedt@ttm' , date)
    
class tax_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('tax@ttm' , date)

class roe_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净资产收益率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('npro@ttm / equ@ttm' , date)
    
class gp_margin_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('gp@ttm / sales@ttm' , date)

class oper_margin_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM营业利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('oper_np@ttm / sales@ttm' , date)

class net_margin_accv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净利率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_accv('npro@ttm / sales@ttm' , date)