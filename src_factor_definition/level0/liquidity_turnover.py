import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.transform import apply_ols

def turnover_classic(date , n_months : int , lag_months : int = 0 , min_finite_ratio = 0.25):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    turns = TSData.TRADE.mask_min_finite(TSData.TRADE.get_turnovers(start_date , end_date , turnover_type = 'fr') , 
                                         min_finite_ratio = min_finite_ratio)
    turns = turns.mean()
    return turns

class turn_1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 1)

class turn_2m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '2个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 2)

class turn_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '3个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 3)
    
class turn_6m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '6个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 6)
    
class turn_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '12个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 12)
    
class turn_unexpected(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月意外换手率'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 3 , 'm' , 1)
        x = TSData.TRADE.get_market_amount(start_date, end_date) / 1e8
        y = TSData.TRADE.get_turnovers(start_date, end_date)

        coef = pd.DataFrame(apply_ols(x , y) , index = ['intercept' , 'slope'] , columns = y.columns)

        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        x = TSData.TRADE.get_market_amount(start_date, end_date) / 1e8
        y = TSData.TRADE.get_turnovers(start_date, end_date)

        pred = pd.DataFrame(x.values * coef.loc[['slope'],:].values + coef.loc[['intercept'],:].values , 
                            index = x.index , columns = coef.columns)
        excess = TSData.TRADE.mask_min_finite(y - pred , min_finite_ratio = 0.25)
        return excess.mean()