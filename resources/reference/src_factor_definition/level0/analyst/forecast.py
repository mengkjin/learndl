import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

__all__ = [
    'etop_est' , 'etop_est_pct3m' , 'etop_est_pct6m' , 
    'eps_est_pct3m' , 'eps_est_pct6m' , 
    'eps_ftm_pct3m' , 'eps_ftm_pct6m' , 
    'optop_est' , 'optop_est_pct3m' , 'optop_est_pct6m' , 
    'epg_est' , 'roe_est' , 
    'sales_cagr1y_est' , 'sales_cagr2y_est' , 
    'npro_cagr1y_est' , 'npro_cagr2y_est' , 
    'stop_est' , 'stop_est_pct3m' , 'stop_est_pct6m' , 
    'tptop_est' , 'tptop_est_pct3m' , 'tptop_est_pct6m' , 
    'price_potential'
]

def val_cagr(date : int , val : Literal['sales' , 'np' , 'tp' , 'op'] , forward_years : Literal[1,2] = 1 , 
             n_month : int = 12 , lag_month : int = 0):
    year = date // 10000
    month = date // 100 % 100
    real_year = (year - 1) if month >= 5 else (year - 2)
    future_year = real_year + forward_years

    real_col = {'sales' : 'revenue' , 'np' : 'n_income_attr_p' , 'tp' : 'total_profit' , 'op' : 'operate_profit'}[val]
    real_val = DATAVENDOR.IS.acc(real_col , date , 2 , year_only=True).reset_index()
    real_val = real_val[real_val['end_date'] == real_year * 10000 + 1231].set_index('secid')[real_col]

    est_val = DATAVENDOR.ANALYST.get_val_est(date , future_year , val , n_month , lag_month)
    cagr = (est_val - real_val) / real_val.abs() / forward_years
    return cagr

def valtop_estimate(date : int , year : int , val : Literal['sales' , 'np' , 'tp' , 'op'] , n_month : int = 12 , lag_month : int = 0):
    v = DATAVENDOR.ANALYST.get_val_est(date , year , val , n_month , lag_month) * 1e4
    td = DATAVENDOR.td(DATAVENDOR.CALENDAR.cd(date , -30 * lag_month))
    mv = DATAVENDOR.TRADE.get_val(td).set_index('secid')['total_mv']
    return v / mv

def valtop_ftm(date : int , val : Literal['sales' , 'np' , 'tp' , 'op'] , n_month : int = 12 , lag_month : int = 0):
    v = DATAVENDOR.ANALYST.get_val_ftm(date , val , n_month , lag_month)
    td = DATAVENDOR.td(DATAVENDOR.CALENDAR.cd(date , -30 * lag_month))
    mv = DATAVENDOR.TRADE.get_val(td).set_index('secid')['total_mv']
    return v / mv

def val_pct(date : int , val : Literal['eps' , 'np' , 'op' , 'tp' , 'sales' , 'roe'] , pct_month : int , 
            ftm = False , n_month : int = 12):
    if ftm:
        v0 = DATAVENDOR.ANALYST.get_val_ftm(date , val , n_month)
        v1 = DATAVENDOR.ANALYST.get_val_ftm(date , val , n_month , pct_month)
    else:
        v0 = DATAVENDOR.ANALYST.get_val_est(date , date // 10000 , val , n_month)
        v1 = DATAVENDOR.ANALYST.get_val_est(date , date // 10000 , val , n_month , pct_month)
    return (v0 - v1) / v1.abs()

def valtop_pct(date : int , val : Literal['sales' , 'np' , 'op' , 'tp'] , pct_month : int , 
               ftm = False , n_month : int = 12):
    if ftm:
        v0 = valtop_ftm(date , val , n_month)
        v1 = valtop_ftm(date , val , n_month , pct_month)
    else:
        v0 = valtop_estimate(date , date // 10000 , val , n_month)
        v1 = valtop_estimate(date , date // 10000 , val , n_month , pct_month)
    return (v0 - v1) / v1.abs()

class etop_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '归母净利润/市值一致预期值'

    def calc_factor(self, date: int):
        return valtop_ftm(date , 'np')
    
class etop_est_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月归母净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'np' , 3)
    
class etop_est_pct6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月归母净利润/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'np' , 6)

class eps_est_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月EPS一致预期变化'

    def calc_factor(self, date: int):
        return val_pct(date , 'eps' , 3)
    
class eps_est_pct6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月EPS一致预期变化'

    def calc_factor(self, date: int):
        return val_pct(date , 'eps' , 6)

class eps_ftm_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月EPSFTM一致预期值变化'

    def calc_factor(self, date: int):
        return val_pct(date , 'eps' , 3 , ftm = True)
    
class eps_ftm_pct6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月EPSFTM一致预期值变化'

    def calc_factor(self, date: int):
        return val_pct(date , 'eps' , 6 , ftm = True)

class optop_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '营业利润/市值一致预期值'

    def calc_factor(self, date: int):
        return valtop_ftm(date , 'op' , 12)
    
class optop_est_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月营业利润/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'op' , 3)
    
class optop_est_pct6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月营业利润/市值一致预期变化'

    UPDATE_RELAX_DATES : list[int] = [20110104 , 20110111 , 20110118]

    def calc_factor(self, date: int):
        return valtop_pct(date , 'op' , 6)

class sales_cagr1y_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '1年营业收入CAGR一致预期值'

    def calc_factor(self, date: int):
        return val_cagr(date , 'sales' , 1 , 12)
    
class sales_cagr2y_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '2年营业收入CAGR一致预期值'

    def calc_factor(self, date: int):
        return val_cagr(date , 'sales' , 2 , 12)
    
class npro_cagr1y_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '1年归母净利润CAGR一致预期值'

    def calc_factor(self, date: int):
        return val_cagr(date , 'np' , 1 , 12)
    
class npro_cagr2y_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '2年归母净利润CAGR一致预期值'

    def calc_factor(self, date: int):
        return val_cagr(date , 'np' , 2 , 12)
    
class epg_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = 'PEG一致预期值'

    def calc_factor(self, date: int):
        cagr = val_cagr(date , 'np' , 1 , 12)
        ep = valtop_estimate(date , date // 10000 , 'np' , 12)
        return ep * cagr
    
class roe_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '净资产收益率一致预期值'

    def calc_factor(self, date: int):
        return DATAVENDOR.ANALYST.get_val_ftm(date , 'roe' , 12)

class stop_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '营业收入/市值一致预期值'

    def calc_factor(self, date: int):
        return valtop_ftm(date , 'sales' , 12)
    
class stop_est_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月营业收入/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'sales' , 3)

class stop_est_pct6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月营业收入/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'sales' , 6)

class tptop_est(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '利润总额/市值一致预期值'

    def calc_factor(self, date: int):
        return valtop_ftm(date , 'tp' , 12)
    
class tptop_est_pct3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '3个月利润总额/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'tp' , 3)

class tptop_est_pct6m(StockFactorCalculator):   
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '6个月利润总额/市值一致预期变化'

    def calc_factor(self, date: int):
        return valtop_pct(date , 'tp' , 6)
    
class price_potential(StockFactorCalculator):
    init_date = 20110101
    category0 = 'analyst'
    category1 = 'forecast'
    description = '分析师评级一致目标价空间'

    def calc_factor(self, date: int):
        tp = DATAVENDOR.ANALYST.target_price(date , 12)
        cp = DATAVENDOR.TRADE.get_trd(date).set_index('secid')['close']
        return (tp - cp) / cp
