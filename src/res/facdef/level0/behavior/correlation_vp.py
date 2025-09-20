import numpy as np

from typing import Literal
from src.data import DATAVENDOR
from src.res.factor.calculator import CorrelationFactor


def vp_correlation(date , n_months : int , volume_type : Literal['amount' , 'volume' , 'turn_tt' , 'turn_fl' , 'turn_fr'] = 'volume' ,
                   price_type : Literal['open' , 'high' , 'close' , 'low' , 'vwap'] = 'close' ,
                   min_finite_ratio = 0.25):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm')
    
    volume = DATAVENDOR.TRADE.get_quotes(start_date, end_date , volume_type , pivot = True)
    price = DATAVENDOR.TRADE.get_quotes(start_date, end_date , price_type , mask=True , pivot = True)

    corr = price.corrwith(volume)
    mask = np.isfinite(price).sum() < len(price) * min_finite_ratio
    corr += (mask * np.nan).where(mask , 0)

    return price.corrwith(volume)

class turnvp_corr1m(CorrelationFactor):
    init_date = 20110101
    description = '1个月换手率-成交均价相关系数'
    
    def calc_factor(self, date: int):
        return vp_correlation(date , 1 , 'turn_fr' , 'vwap')
    
class turnvp_corr2m(CorrelationFactor):
    init_date = 20110101
    description = '2个月换手率-成交均价相关系数'
    
    def calc_factor(self, date: int):
        return vp_correlation(date , 2 , 'turn_fr' , 'vwap')
    
class turnvp_corr3m(CorrelationFactor):
    init_date = 20110101
    description = '3个月换手率-成交均价相关系数'
    
    def calc_factor(self, date: int):
        return vp_correlation(date , 3 , 'turn_fr' , 'vwap')
    
class turnvp_corr6m(CorrelationFactor):
    init_date = 20110101
    description = '6个月换手率-成交均价相关系数'
    
    def calc_factor(self, date: int):
        return vp_correlation(date , 6 , 'turn_fr' , 'vwap')
    
class turnvp_corr12m(CorrelationFactor):
    init_date = 20110101
    description = '12个月换手率-成交均价相关系数'
    
    def calc_factor(self, date: int):
        return vp_correlation(date , 12 , 'turn_fr' , 'vwap')