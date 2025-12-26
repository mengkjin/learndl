import pandas as pd

from src.data import DATAVENDOR
from src.res.factor.calculator import LiquidityFactor

from src.math.transform import apply_ols

def turnover_classic(date , n_months : int , lag_months : int = 0 , min_finite_ratio = 0.25):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    turns = DATAVENDOR.TRADE.mask_min_finite(DATAVENDOR.TRADE.get_turnovers(start_date , end_date , turnover_type = 'fr') , 
                                         min_finite_ratio = min_finite_ratio)
    turns = turns.mean()
    return turns

class turn_1m(LiquidityFactor):
    init_date = 20110101
    description = '1个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 1)

class turn_2m(LiquidityFactor):
    init_date = 20110101
    description = '2个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 2)

class turn_3m(LiquidityFactor):
    init_date = 20110101
    description = '3个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 3)
    
class turn_6m(LiquidityFactor):
    init_date = 20110101
    description = '6个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 6)
    
class turn_12m(LiquidityFactor):
    init_date = 20110101
    description = '12个月换手率'
    
    def calc_factor(self, date: int):
        return turnover_classic(date , 12)
    
class turn_unexpected(LiquidityFactor):
    init_date = 20110101
    description = '1个月意外换手率'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 3 , 'm' , 1)
        x = DATAVENDOR.TRADE.get_market_amount(start_date, end_date) / 1e8
        y = DATAVENDOR.TRADE.get_turnovers(start_date, end_date)

        coef = pd.DataFrame(apply_ols(x , y) , index = pd.Index(['intercept' , 'slope']) , columns = y.columns)

        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        x = DATAVENDOR.TRADE.get_market_amount(start_date, end_date) / 1e8
        y = DATAVENDOR.TRADE.get_turnovers(start_date, end_date)

        pred = pd.DataFrame(x.values * coef.loc[['slope'],:].values + coef.loc[['intercept'],:].values , 
                            index = x.index , columns = coef.columns)
        excess = DATAVENDOR.TRADE.mask_min_finite(y - pred , min_finite_ratio = 0.25)
        return excess.mean()