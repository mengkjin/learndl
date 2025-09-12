import numpy as np
import pandas as pd

from typing import Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'btop_stability' ,
    'dtop_stability' ,
    'etop_stability' ,
    'stop_stability'
]

def calc_stability(numerator : pd.DataFrame | pd.Series | float | int , denominator : pd.DataFrame | pd.Series | float | int , 
                   reindex_like : Literal['numerator' , 'denominator'] = 'denominator'):
    if isinstance(numerator , pd.DataFrame) and isinstance(denominator , pd.DataFrame):
        union_index = numerator.index.union(denominator.index).sort_values()
        if reindex_like == 'numerator':
            denominator = denominator.reindex(index = union_index).ffill().reindex_like(numerator)
        else:
            numerator = numerator.reindex(index = union_index).ffill().reindex_like(denominator)
    ratio = numerator / denominator
    assert isinstance(ratio , pd.DataFrame) , type(ratio)
    valid = ratio.notna().sum() > 12
    return (ratio.mean() / ratio.std()).where(valid , np.nan)

def get_ev_hist(date: int , n_year : int = 1 , date_step : int = 1):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_year , 'y')
    mv   = DATAVENDOR.TRADE.get_specific_data(start_date , end_date , 'val' , 'total_mv' , 
                                              prev = False , pivot = True , date_step = date_step)
    liab = DATAVENDOR.BS.acc('total_liab'   , date , n_year * 4 + 1 , pivot = True)
    debt = DATAVENDOR.BS.acc('bond_payable' , date , n_year * 4 + 1 , pivot = True)
    cash = DATAVENDOR.BS.acc('money_cap'    , date , n_year * 4 + 1 , pivot = True)
    
    union_index = mv.index.union(debt.index).sort_values()
    liab = liab.reindex(index = union_index).ffill().reindex_like(mv).fillna(0)
    debt = debt.reindex(index = union_index).ffill().reindex_like(mv).fillna(0)
    cash = cash.reindex(index = union_index).ffill().reindex_like(mv).fillna(0)

    indus = DATAVENDOR.INFO.get_indus(date).iloc[:,0].reindex(mv.columns)
    added = debt
    added.loc[: , indus == 'bank'] = (liab - cash).loc[: , indus == 'bank']

    ev = mv + added
    return ev

def get_denominator_hist(denominator : Literal['mv' , 'ev'] | str , date : int , n_year = 3 , date_step = 21):
    if denominator == 'mv':
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_year , 'y')
        v = DATAVENDOR.TRADE.get_specific_data(start_date , end_date , 'val' , 'total_mv' , 
                                               prev = False , pivot = True , date_step = date_step)
    elif denominator == 'ev':
        v = get_ev_hist(date , n_year , date_step)
    else:
        raise KeyError(denominator)
    return v

def valuation_stability(numerator : str , date : int):
    num = DATAVENDOR.get_fin_hist(numerator , date , 3 * 4 + 1 , pivot = True)
    den = get_denominator_hist('mv' , date , 3)
    return calc_stability(num , den)

class btop_stability(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'BP稳定性,均值除标准差'
    
    def calc_factor(self, date: int):
        return valuation_stability('equ@qtr' , date)

class dtop_stability(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'DP稳定性,均值除标准差'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 3 , 'y')
        dv_ttm = DATAVENDOR.TRADE.get_specific_data(
            start_date , end_date , 'val' , 'dv_ttm' , pivot = True , date_step = 21)
        return calc_stability(dv_ttm , 1)

class etop_stability(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'ETOP稳定性,均值除标准差'
    
    def calc_factor(self, date: int):
        return valuation_stability('npro@qtr' , date)
    
class stop_stability(StockFactorCalculator):    
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'STOP稳定性,均值除标准差'
    
    def calc_factor(self, date: int):
        return valuation_stability('sales@qtr' , date)
    
class cfev_stability(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'CFEV稳定性,均值除标准差'
    
    def calc_factor(self, date: int):
        return valuation_stability('ncfo@qtr' , date)