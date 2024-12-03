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
        eps = DATAVENDOR.INDI.qtr('eps' , date , 10 , True , ffill=True)
        eps = eps.groupby('secid').tail(5)['eps']
        valid = eps.groupby('secid').size() > 4
        delta_eps = eps.groupby('secid').tail(1) - eps.groupby('secid').head(1)
        base = eps.groupby('secid').head(1).abs()
        df = (delta_eps / base).where(valid , np.nan)
        return df
    
class acc_eav(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '盈利加速(Earnings acceleration),除以最近8个季度EPS的标准差'
    
    def calc_factor(self, date: int):
        eps = DATAVENDOR.INDI.qtr('eps' , date , 10 , True , ffill=True)
        eps = eps.groupby('secid').tail(8)['eps']
        valid = eps.groupby('secid').size() > 4
        delta_eps = eps.groupby('secid').tail(1) - eps.groupby('secid').nth(4)
        base = eps.groupby('secid').std()
        df = (delta_eps / base).where(valid , np.nan)
        return df