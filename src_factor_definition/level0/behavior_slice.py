import numpy as np
import pandas as pd

from typing import Literal
from src.factor.classes import StockFactorCalculator
from src.data import DATAVENDOR
from src.func.transform import neutral_resid

def get_amplitudes(start_date , end_date , pivot = True):
    quotes = DATAVENDOR.TRADE.get_quotes(start_date , end_date , ['high' , 'low' , 'preclose'] , adj_price = False)
    amplitudes = ((quotes['high'] - quotes['low']) / quotes['preclose']).rename('amplitude').\
        to_frame()
    if pivot: amplitudes = amplitudes.pivot_table('amplitude' , 'date' , 'secid').fillna(0)
    return amplitudes

def get_slicing(start_date , end_date , sliced_by : Literal['amplitude' , 'vol' , 'cp'] , slice_ratio = 0.5):
    if sliced_by == 'amplitude':
        slice_values = get_amplitudes(start_date , end_date , pivot = True)
    elif sliced_by == 'vol':
        slice_values =  DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'volume' , pivot = True)
    elif sliced_by == 'cp':
        slice_values =  DATAVENDOR.TRADE.get_quotes(start_date , end_date , 'close' , pivot = True , adj_price = False)
    sliced_pct = slice_values.rank(axis = 0 , pct = True , method = 'first') 
    slicing = sliced_pct >= slice_ratio 
    return slicing , ~slicing

def mom_low_amp_v2(date , n_days : int , low_amplitude_ratio = 0.7):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_days , 'd' , 0)

    quotes = DATAVENDOR.TRADE.get_quotes(start_date , end_date , ['pctchange' , 'high' , 'low' , 'preclose' , 'close' , 'status' , 'limit'] , adj_price = False)
    mkt_ret = DATAVENDOR.TRADE.get_market_return(start_date , end_date)

    quotes = quotes.join(mkt_ret , on = ['date'] , how = 'left')

    quotes['ignore'] = ((quotes['limit'] != 0) + (quotes['status'] == 0)).fillna(True)
    quotes['amplitude'] = ((quotes['high'] - quotes['low']) / quotes['preclose']).where(~quotes['ignore'] , np.nan)
    quotes['alpha'] = quotes['pctchange'] / 100 - quotes['market']

    pivoted = quotes.pivot_table(['alpha' , 'amplitude'] , 'date' , 'secid')

    lamp = pivoted['amplitude'].rank(axis = 0 , pct = True , method = 'first') <= low_amplitude_ratio
    mom = pivoted['alpha'].where(lamp , np.nan).sum()

    ret20 = ((1 + DATAVENDOR.TRADE.get_returns(*DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd') , pivot=True)).prod() - 1).reindex(mom.index)

    mom = neutral_resid(ret20 , mom , whiten = False)
    return mom

class mom_ltampl_v1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '160日长端动量(低振幅)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 160 , 'd' , 0)
        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True)
        h , l = get_slicing(start_date , end_date , 'amplitude' , 0.7)
        mom = rets.where(l , np.nan).sum()
        return mom
    
class mom_ltampl_v2(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '160日长端动量(低振幅)v2'
    
    def calc_factor(self, date: int):
        return mom_low_amp_v2(date , 160 , 0.7)
    
class mom_slicevol1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月理想反转(成交量切分，较大-较小)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd' , 0)
        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True)
        h , l = get_slicing(start_date , end_date , 'vol')
        mom = rets.where(h , np.nan).sum() - rets.where(l , np.nan).sum()
        return mom
    
class corr_slicevol1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '1个月corr差(成交量切分，较大-较小)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd' , 0)
        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True)
        market_rets = DATAVENDOR.TRADE.get_market_return(start_date , end_date)
        h , l = get_slicing(start_date , end_date , 'vol')
        h_stk , h_mkt = rets.where(h , np.nan) , (h.T * market_rets['market']).T.where(h , np.nan)
        l_stk , l_mkt = rets.where(l , np.nan) , (l.T * market_rets['market']).T.where(l , np.nan)
        diff = h_stk.corrwith(h_mkt , axis = 0) - l_stk.corrwith(l_mkt , axis = 0)
        return diff
    
class beta_slicevol1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'correlation'
    description = '1个月beta差(成交量切分，较大-较小)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd' , 0)
        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True)
        market_rets = DATAVENDOR.TRADE.get_market_return(start_date , end_date)
        h , l = get_slicing(start_date , end_date , 'vol')
        h_stk , h_mkt = rets.where(h , np.nan) , (h.T * market_rets['market']).T.where(h , np.nan)
        l_stk , l_mkt = rets.where(l , np.nan) , (l.T * market_rets['market']).T.where(l , np.nan)
        diff = h_stk.corrwith(h_mkt , axis = 0) * h_stk.std() / h_mkt.std() - l_stk.corrwith(l_mkt , axis = 0) * l_stk.std() / l_mkt.std()
        return diff

class skew_slicevol1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月skew差(成交量切分，较大-较小)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd' , 0)
        rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True)
        h , l = get_slicing(start_date , end_date , 'vol')
        h_rets , l_rets = rets.where(h , np.nan) , rets.where(l , np.nan)
        diff = h_rets.skew() - l_rets.skew()
        return diff

class ampl_slicecp1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = '1个月振幅(收盘价切分，较大)'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 20 , 'd' , 0)
        ampl = get_amplitudes(start_date , end_date , pivot = True)
        h , l = get_slicing(start_date , end_date , 'cp')
        ampl = ampl.where(h , np.nan).mean()
        return ampl