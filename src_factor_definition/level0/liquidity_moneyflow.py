import numpy as np
import pandas as pd
from typing import Literal

from src.factor.classes import StockFactorCalculator
from src.data import TSData

def money_inflow(date , n_months : int , direction : Literal[-1,1] , div_amt = True ,
                 min_finite_ratio = 0.25):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm')
    net_mfs   = TSData.TRADE.get_mf_data(start_date , end_date , 'net_mf_amount' , pivot = True)
    rets_sign = np.sign(TSData.TRADE.get_returns(start_date , end_date , pivot = True))

    net_mfs *= 1 * (rets_sign == direction) + 0.5 * (rets_sign == 0)
    inflow = net_mfs.sum()
    inflow[np.isfinite(net_mfs).sum() < len(net_mfs) * min_finite_ratio] = np.nan
    if div_amt:
        inflow /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True).sum()
    return inflow
    
class bsact_neg_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流入-负收益'
    
    def calc_factor(self, date: int):
        return money_inflow(date , 1 , -1)
    
class bsact_pos_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流入-正收益'
    
    def calc_factor(self, date: int):
        return money_inflow(date , 1 , 1)