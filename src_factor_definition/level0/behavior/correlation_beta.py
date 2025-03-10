import numpy as np
import pandas as pd
from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR
from src.func.transform import time_weight , apply_ols

def calc_beta(date , n_months : int , lag_months : int = 0 , half_life = 0 , min_finite_ratio = 0.25):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)

    stk_ret = DATAVENDOR.TRADE.get_returns(start_date, end_date)
    mkt_ret = DATAVENDOR.TRADE.get_market_return(start_date, end_date)
    wgt = time_weight(len(mkt_ret) , half_life)
    beta = pd.Series(apply_ols(mkt_ret.values.flatten() , stk_ret.values , wgt)[1] , index = stk_ret.columns)
    mask = np.isfinite(stk_ret).sum() < len(mkt_ret) * min_finite_ratio
    beta += (mask * np.nan).where(mask , 0)

    return beta

class beta_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '1个月贝塔'
    
    def calc_factor(self, date: int):
        return calc_beta(date , 1)
    
class beta_2m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '2个月贝塔'
    
    def calc_factor(self, date: int):
        return calc_beta(date , 2)

class beta_3m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '3个月贝塔'
    
    def calc_factor(self, date: int):
        return calc_beta(date , 3)

class beta_6m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '6个月贝塔'
    
    def calc_factor(self, date: int):
        return calc_beta(date , 6)
    
class beta_12m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '12个月贝塔'
    
    def calc_factor(self, date: int):
        return calc_beta(date , 12)