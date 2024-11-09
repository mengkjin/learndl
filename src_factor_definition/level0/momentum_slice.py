import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.transform import neutral_resid

def mom_low_amp(date , n_days : int , low_amplitude_ratio = 0.7):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_days , 'd' , 0)

    quotes = TSData.TRADE.get_quotes(start_date , end_date , ['pctchange' , 'high' , 'low' , 'preclose'] , adj_price = False)
    rets = quotes.pivot_table('pctchange' , 'date' , 'secid') / 100
    amplitudes = ((quotes['high'] - quotes['low']) / quotes['preclose']).rename('amplitude').\
        to_frame().pivot_table('amplitude' , 'date' , 'secid').fillna(0)
    lamp = amplitudes.rank(axis = 0 , pct = True , method = 'first') <= low_amplitude_ratio
    mom = (rets.where(lamp , np.nan) + 1).prod() - 1
    return mom

def mom_low_amp_v2(date , n_days : int , low_amplitude_ratio = 0.7):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_days , 'd' , 0)

    quotes = TSData.TRADE.get_quotes(start_date , end_date , ['pctchange' , 'high' , 'low' , 'preclose' , 'close' , 'status' , 'limit'] , adj_price = False)
    mkt_ret = TSData.TRADE.get_market_return(start_date , end_date)

    quotes = quotes.join(mkt_ret , on = ['date'] , how = 'left')

    quotes['ignore'] = ((quotes['limit'] != 0) + (quotes['status'] == 0)).fillna(True)
    quotes['amplitude'] = ((quotes['high'] - quotes['low']) / quotes['preclose']).where(~quotes['ignore'] , np.nan)
    quotes['alpha'] = quotes['pctchange'] / 100 - quotes['market']

    pivoted = quotes.pivot_table(['alpha' , 'amplitude'] , 'date' , 'secid')

    lamp = pivoted['amplitude'].rank(axis = 0 , pct = True , method = 'first') <= low_amplitude_ratio
    mom = pivoted['alpha'].where(lamp , np.nan).sum()

    ret20 = ((1 + TSData.TRADE.get_returns(*TSData.CALENDAR.td_start_end(date , 20 , 'd') , pivot=True)).prod() - 1).reindex(mom.index)

    mom = neutral_resid(ret20 , mom , whiten = False)
    return mom

class mom_ltampl_v1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '160日长端动量(低振幅)'
    
    def calc_factor(self, date: int):
        return mom_low_amp(date , 160 , 0.7)
    
class mom_ltampl_v2(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '160日长端动量(低振幅)v2'
    
    def calc_factor(self, date: int):
        return mom_low_amp_v2(date , 160 , 0.7)