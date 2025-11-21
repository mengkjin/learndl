import pandas as pd

from src.data import DATAVENDOR
from src.res.factor.calculator import VolatilityFactor

def atr_classic(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    quotes = DATAVENDOR.TRADE.get_quotes(start_date , end_date , ['high' , 'low' , 'close' , 'preclose'])
    quotes['tr'] = pd.concat([
        quotes['high'] - quotes['low'] ,
        abs(quotes['high'] - quotes['preclose']) ,
        abs(quotes['low'] - quotes['preclose']) ,
    ] , axis = 1).max(axis = 1) / quotes['preclose']
    return quotes.groupby('secid')['tr'].mean()

class atr_1m(VolatilityFactor):
    init_date = 20110101
    description = '1个月日内真实波幅均值'
    preprocess = False
    
    def calc_factor(self, date: int):
        return atr_classic(date , 1)