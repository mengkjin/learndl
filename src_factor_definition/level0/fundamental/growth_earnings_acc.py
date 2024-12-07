import numpy as np
import pandas as pd

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

class acc_eaa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '盈利加速(Earnings acceleration),除以去年同期EPS的绝对值'
    
    def calc_factor(self, date: int):
        eps = DATAVENDOR.INDI.qtr('eps' , date , 10 , False)
        delta1 = (eps - eps.groupby('secid').shift(4)) / eps.groupby('secid').shift(4).abs()
        delta2 = (eps.groupby('secid').shift(1) - eps.groupby('secid').shift(5)) / eps.groupby('secid').shift(5).abs()
        valid = eps.groupby('secid').size() > 6
        df = (delta1 - delta2).groupby('secid').last().where(valid , np.nan).iloc[:,0]
        return df
    
class acc_eav(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '盈利加速(Earnings acceleration),除以最近8个季度EPS的标准差'
    
    def calc_factor(self, date: int):
        eps = DATAVENDOR.INDI.qtr('eps' , date , 10 , False)
        eps1 = eps.groupby('secid').tail(8)
        eps2 = eps.groupby('secid').shift(1).groupby('secid').tail(8)
        delta1 = (eps1 - eps1.groupby('secid').shift(4)) / eps1.groupby('secid').std()
        delta2 = (eps2 - eps2.groupby('secid').shift(4)) / eps2.groupby('secid').std()
        valid = eps.groupby('secid').size() > 6
        df = (delta1 - delta2).groupby('secid').last().where(valid , np.nan).iloc[:,0]
        return df