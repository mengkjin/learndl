import numpy as np
import pandas as pd

from typing import Any , Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'btop' , 'btop_rank1y' , 'btop_rank3y' , 
    'dtop' , 'dtop_rank1y' , 'dtop_rank3y' , 
    'ebitev_ttm' , 'ebitev_ttm_rank1y' , 'ebitev_ttm_rank3y' , 
    'ebitdaev_ttm' , 'ebitdaev_ttm_rank1y' , 'ebitdaev_ttm_rank3y' , 
    'etop_ttm' , 'etop_ttm_rank1y' , 'etop_ttm_rank3y' , 
    'etop_dedt_ttm' , 'etop_dedt_ttm_rank1y' , 'etop_dedt_ttm_rank3y' , 
    'etop' , 'etop_rank1y' , 'etop_rank3y' , 
    'fcfetop_ttm' , 'fcfetop_ttm_rank1y' , 'fcfetop_ttm_rank3y' , 
    'fcfetop_qtr' , 'fcfetop_qtr_rank1y' , 'fcfetop_qtr_rank3y' , 
    'cfotop_ttm' , 'cfotop_ttm_rank1y' , 'cfotop_ttm_rank3y' , 
    'cfotop' , 'cfotop_rank1y' , 'cfotop_rank3y' , 
    'stop_ttm' , 'stop_ttm_rank1y' , 'stop_ttm_rank3y' , 
    'stop' , 'stop_rank1y' , 'stop_rank3y' , 
    'cfev' , 'cfev_rank1y' , 'cfev_rank3y' , 
    'cfev_ttm' , 'cfev_ttm_rank1y' , 'cfev_ttm_rank3y' , 
]

def calc_valuation(numerator : pd.DataFrame | pd.Series | float | int , denominator : pd.DataFrame | pd.Series | float | int , 
                   pct = True , reindex_like : Literal['numerator' , 'denominator'] = 'denominator'):
    if isinstance(numerator , pd.DataFrame) and isinstance(denominator , pd.DataFrame):
        union_index = numerator.index.union(denominator.index).sort_values()
        if reindex_like == 'numerator':
            denominator = denominator.reindex(index = union_index).ffill().reindex_like(numerator)
        else:
            numerator = numerator.reindex(index = union_index).ffill().reindex_like(denominator)
    ratio = numerator / denominator
    if isinstance(ratio , pd.DataFrame):
        if pct: ratio = ratio.rank(pct=True)
        return ratio.tail(1).iloc[-1]
    elif isinstance(ratio , pd.Series):
        if pct: 
            raise ValueError(f'If valuation is pd.Series, pct is not a proper input')
            ratio = ratio.rank(pct=True)
        return ratio
    else:
        raise TypeError(f'type incorrect: {type(numerator)} , {type(denominator)}')
    
def get_ev(date: int):
    mv = DATAVENDOR.TRADE.get_val(date , ['secid','total_mv']).set_index(['secid'])['total_mv'].sort_index()
    liab = DATAVENDOR.BS.acc_latest('total_liab' , date).reindex_like(mv).fillna(0)
    debt = DATAVENDOR.BS.acc_latest('bond_payable' , date).reindex_like(mv).fillna(0)
    cash = DATAVENDOR.BS.acc_latest('money_cap' , date).reindex_like(mv).fillna(0)
    indus = DATAVENDOR.INFO.get_indus(date).iloc[:,0].reindex(mv)
    added = debt.where(indus == 'bank' , liab - cash)
    ev = mv + added
    return ev

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

def get_denominator(denominator : Literal['mv' , 'cp' , 'ev'] | str , date : int):
    if denominator == 'mv':
        v = DATAVENDOR.TRADE.get_val(date , ['secid','total_mv']).set_index(['secid'])['total_mv'].sort_index()
    elif denominator == 'cp':
        v = DATAVENDOR.TRADE.get_trd(date , ['secid','close']).set_index(['secid'])['close'].sort_index()
    elif denominator == 'ev':
        v = get_ev(date)
    else:
        raise KeyError(denominator)
    return v

def get_denominator_hist(denominator : Literal['mv' , 'cp' , 'ev1'] | str , date : int , n_year = 1 , date_step = 1):
    if denominator == 'mv':
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_year , 'y')
        v = DATAVENDOR.TRADE.get_specific_data(start_date , end_date , 'val' , 'total_mv' , 
                                               prev = False , pivot = True , date_step = date_step)
    elif denominator == 'cp':
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_year , 'y')
        v = DATAVENDOR.TRADE.get_specific_data(start_date , end_date , 'trd' , 'close' , 
                                               prev = False , pivot = True , date_step = date_step)
    elif denominator == 'ev':
        v = get_ev_hist(date , n_year , date_step)
    else:
        raise KeyError(denominator)
    return v

def valuation_latest(numerator : str , denominator , date : int , qtr_method = 'diff'):
    kwargs = {'qtr_method' : qtr_method} if numerator.startswith('indi') else {}
    num = DATAVENDOR.get_fin_latest(numerator , date , **kwargs)
    den = get_denominator(denominator , date)
    return calc_valuation(num , den , pct = False)

def valuation_rank1y(numerator : str , denominator , date : int , qtr_method = 'diff'):
    kwargs = {'qtr_method' : qtr_method} if numerator.startswith('indi') else {}
    num = DATAVENDOR.get_fin_hist(numerator , date , 5 , pivot = True ,**kwargs)
    den = get_denominator_hist(denominator , date , 1 , 1)
    return calc_valuation(num , den , pct = True)

def valuation_rank3y(numerator : str , denominator , date : int , qtr_method = 'diff'):
    kwargs = {'qtr_method' : qtr_method} if numerator.startswith('indi') else {}
    num = DATAVENDOR.get_fin_hist(numerator , date , 13 , pivot = True ,**kwargs)
    den = get_denominator_hist(denominator , date , 3 , 21)
    return calc_valuation(num , den , pct = True)

class btop(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '市净率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('equ@qtr' , 'mv' , date)
    
class btop_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '市净率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('equ@qtr' , 'mv' , date)
    
class btop_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '市净率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('equ@qtr' , 'mv' , date)
    
class dtop(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '滚动分红率'
    
    def calc_factor(self, date: int):
        # some date this column is erronous
        dv_ttm = DATAVENDOR.TRADE.get_val(date).set_index(['secid'])['dv_ttm'] / 100
        while dv_ttm.isna().all():
            start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
            dv_ttm = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'dv_ttm' , prev=False , pivot=False)
            dv_ttm = dv_ttm.dropna().groupby('secid')['dv_ttm'].last() / 100
        return calc_valuation(dv_ttm , 1 , pct = False)
    
class dtop_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '股东分红率,1年分位数'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
        dv_ttm = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'dv_ttm' , prev=False , pivot=True)
        return calc_valuation(dv_ttm , 1 , pct = True)
    
class dtop_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '股东分红率,3年分位数'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 3 , 'y')
        dv_ttm = DATAVENDOR.TRADE.get_specific_data(start_date , end_date , 'val' , 'dv_ttm' , prev=False , pivot=True , date_step = 21)
        return calc_valuation(dv_ttm , 1 , pct = True)

class ebitev_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV'
    
    def calc_factor(self, date: int):
        return valuation_latest('ebit@ttm' , 'ev' , date)

class ebitev_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ebit@ttm' , 'ev' , date)

class ebitev_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ebit@ttm' , 'ev' , date)
    
class ebitdaev_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV'
    
    def calc_factor(self, date: int):
        return valuation_latest('ebitda@ttm' , 'ev' , date)

class ebitdaev_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ebitda@ttm' , 'ev' , date)

class ebitdaev_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ebitda@ttm' , 'ev' , date)
    
class etop_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('npro@ttm' , 'mv' , date)
    
class etop_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市盈率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('npro@ttm' , 'mv' , date)
    
class etop_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市盈率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('npro@ttm' , 'mv' , date)
    
class etop_dedt_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM扣非市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('dedt@ttm' , 'mv' , date)
    
class etop_dedt_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM扣非市盈率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('dedt@ttm' , 'mv' , date)
    
class etop_dedt_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM扣非市盈率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('dedt@ttm' , 'mv' , date)

class etop(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('npro@qtr' , 'mv' , date)

class etop_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市盈率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('npro@qtr' , 'mv' , date)
    
class etop_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市盈率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('npro@qtr' , 'mv' , date)
    
class fcfetop_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM企业股权自由现金流量/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('fcfe@ttm' , 'mv' , date)

class fcfetop_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM企业股权自由现金流量/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('fcfe@ttm' , 'mv' , date)
    
class fcfetop_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM企业股权自由现金流量/市值,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('fcfe@ttm' , 'mv' , date)


class fcfetop_qtr(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度企业股权自由现金流量/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('fcfe@qtr' , 'mv' , date)

class fcfetop_qtr_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度企业股权自由现金流量/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('fcfe@qtr' , 'mv' , date)
    
class fcfetop_qtr_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度企业股权自由现金流量/市值,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('fcfe@qtr' , 'mv' , date)

    
class cfotop_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营现金流/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('ncfo@ttm' , 'mv' , date)

class cfotop_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营现金流/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ncfo@ttm' , 'mv' , date)
    
class cfotop_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营现金流/市值,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ncfo@ttm' , 'mv' , date)

class cfotop(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营现金流/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('ncfo@qtr' , 'mv' , date)

class cfotop_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营现金流/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ncfo@qtr' , 'mv' , date)
    
class cfotop_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营现金流/市值,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ncfo@qtr' , 'mv' , date)
    
class stop_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市销率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('sales@ttm' , 'mv' , date)
    
class stop_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市销率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('sales@ttm' , 'mv' , date)
    
class stop_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市销率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('sales@ttm' , 'mv' , date)

class stop(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市销率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('sales@qtr' , 'mv' , date)

class stop_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市销率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('sales@qtr' , 'mv' , date)
    
class stop_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市销率倒数,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('sales@qtr' , 'mv' , date)
    
class cfev(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营性现金流/EV'
    
    def calc_factor(self, date: int):
        return valuation_latest('ncfo@qtr' , 'ev' , date)
    
class cfev_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营性现金流/EV,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ncfo@qtr' , 'ev' , date)
    
class cfev_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营性现金流/EV,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ncfo@qtr' , 'ev' , date)

    
class cfev_ttm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营性现金流/EV'
    
    def calc_factor(self, date: int):
        return valuation_latest('ncfo@ttm' , 'ev' , date)
    
class cfev_ttm_rank1y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营性现金流/EV,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('ncfo@ttm' , 'ev' , date)
    
class cfev_ttm_rank3y(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营性现金流/EV,3年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank3y('ncfo@ttm' , 'ev' , date)

