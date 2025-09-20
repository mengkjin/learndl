import numpy as np

from src.data import DATAVENDOR
from src.res.factor.calculator import VolatilityFactor


def exret_std(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    exrets = DATAVENDOR.RISK.get_exret(start_date , end_date , pivot=True)
    return exrets.std() * np.sqrt(252)

class exret_std1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 1)
    
class exret_std2m(VolatilityFactor):
    init_date = 20110101
    description = '2个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 2)
 
class exret_std3m(VolatilityFactor):
    init_date = 20110101
    description = '3个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 3)
    
class exret_std6m(VolatilityFactor):
    init_date = 20110101
    description = '6个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 6)
    
class exret_std12m(VolatilityFactor):
    init_date = 20110101
    description = '12个月超额收益率标准差'
    
    def calc_factor(self, date: int):
        return exret_std(date , 12)