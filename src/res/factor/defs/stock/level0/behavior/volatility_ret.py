import numpy as np

from src.data import DATAVENDOR
from src.res.factor.calculator import VolatilityFactor

def ret_std_classic(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    returns = DATAVENDOR.TRADE.get_returns(start_date , end_date)
    return returns.std() * np.sqrt(252)

class ret_std1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 1)
    
class ret_std2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 2)
    
class ret_std3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 3)
    
class ret_std6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 6)
    
class ret_std12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月收益率标准差'
    
    def calc_factor(self, date: int):
        return ret_std_classic(date , 12)