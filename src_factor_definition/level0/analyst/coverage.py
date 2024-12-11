import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'cov_inst_3m' , 'cov_inst_6m' , 'cov_inst_12m' , 'cov_inst_anndt' ,
    'cov_report_3m' , 'cov_report_6m' , 'cov_report_12m' ,
]

class cov_inst_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '3个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_inst_3m')
        return df
    
class cov_inst_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '6个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_inst_6m')
        return df
    
class cov_inst_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '12个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_inst_6m')
        return df
    
class cov_inst_anndt(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '区间内覆盖股票机构数量--公告日'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_inst_anndt')
        return df
    
class cov_report_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '3个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_report_3m')
        return df
    
class cov_report_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '6个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_report_6m')
        return df
    
class cov_report_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '12个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_data('analyst_forecast_coverage' , date , 'cov_report_12m')
        return df