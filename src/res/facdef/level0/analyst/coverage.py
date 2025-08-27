import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator

__all__ = [
    'cov_inst_3m' , 'cov_inst_6m' , 'cov_inst_12m' , 'cov_inst_12m_anndt' ,
    'cov_report_3m' , 'cov_report_6m' , 'cov_report_12m' ,
]

def report_inst_count(date : int , n_month: int):
    secid = DATAVENDOR.secid(date)
    df = DATAVENDOR.ANALYST.get_trailing_reports(date , n_month).groupby('secid')['org_name'].nunique()
    return df.reindex(secid).fillna(0)

def report_report_count(date : int , n_month: int):
    secid = DATAVENDOR.secid(date)
    df = DATAVENDOR.ANALYST.get_trailing_reports(date , n_month).reset_index()[['secid','org_name','report_date','report_title']]
    df = df.drop_duplicates().groupby('secid').size()
    return df.reindex(secid).fillna(0)

class cov_inst_3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '3个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        return report_inst_count(date, 3)
    
class cov_inst_6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '6个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        return report_inst_count(date, 6)
class cov_inst_12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '12个月区间内覆盖股票机构数量'
    
    def calc_factor(self, date: int):
        return report_inst_count(date, 12)
    
class cov_inst_12m_anndt(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '12个月区间内覆盖股票机构数量,公告日后7天'
    
    def calc_factor(self, date: int):
        secid  = DATAVENDOR.secid(date)
        df      = DATAVENDOR.ANALYST.get_trailing_reports(date , 12).set_index(['secid','report_date'])
        ann_cal = DATAVENDOR.IS.get_ann_calendar(date , after_days = 7 , within_days = 365).\
            rename_axis(index = {'date':'report_date'}).reindex(df.index)
        df = df[ann_cal['anndt'] > 0].groupby('secid')['org_name'].nunique().reindex(secid).fillna(0)
        return df
    
class cov_report_3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '3个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        return report_report_count(date , 3)
    
class cov_report_6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '6个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        return report_report_count(date , 6)
    
class cov_report_12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'coverage'
    description = '12个月区间内覆盖股票报告数量'
    
    def calc_factor(self, date: int):
        return report_report_count(date , 12)