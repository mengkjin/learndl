import numpy as np
import pandas as pd

from typing import Any , Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

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
    
def get_ev1(date: int):
    mv = DATAVENDOR.TRADE.get_val(date , ['secid','total_mv']).set_index(['secid'])['total_mv'].sort_index()
    debt = DATAVENDOR.INDI.acc_latest('interestdebt' , date)
    ev = mv + debt
    return ev

def get_ev2(date: int):
    ev1 = get_ev1(date)
    cash = DATAVENDOR.BS.acc_latest('money_cap' , date)
    ev2 = ev1 - cash
    return ev2

def get_ev1_hist(date: int , n_year : int = 1):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_year , 'y')
    mv   = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'total_mv' , prev=False , pivot=True)
    debt = DATAVENDOR.INDI.acc('interestdebt' , date , n_year * 4 + 1 , pivot = True)
    union_index = mv.index.union(debt.index).sort_values()
    debt = debt.reindex(index = union_index).ffill().reindex_like(mv).fillna(0)
    ev = mv + debt
    return ev

def get_ev2_hist(date: int , n_year : int = 1):
    ev1  = get_ev1_hist(date , n_year)
    cash = DATAVENDOR.BS.acc('money_cap' , date , n_year * 4 + 1 , pivot = True)
    union_index = ev1.index.union(cash.index).sort_values()
    cash = cash.reindex(index = union_index).ffill().reindex_like(ev1).fillna(0)
    ev2 = ev1 - cash
    return ev2

def get_numerator(numerator : str , date : int , **kwargs):
    return DATAVENDOR.get_fin_latest(numerator , date , **kwargs)

def get_numerator_hist(numerator : str , date : int , n_year : int = 1 , **kwargs):
    return DATAVENDOR.get_fin_hist(numerator , date , n_year * 4 + 1 , pivot = True ,**kwargs)

def get_denominator(denominator : Literal['mv' , 'cp' , 'ev1' , 'ev2'] | str , date : int):
    if denominator == 'mv':
        v = DATAVENDOR.TRADE.get_val(date , ['secid','total_mv']).set_index(['secid'])['total_mv'].sort_index()
    elif denominator == 'cp':
        v = DATAVENDOR.TRADE.get_trd(date , ['secid','close']).set_index(['secid'])['close'].sort_index()
    elif denominator == 'ev1':
        v = get_ev1(date)
    elif denominator == 'ev2':
        v = get_ev2(date)
    else:
        raise KeyError(denominator)
    return v

def get_denominator_hist(denominator : Literal['mv' , 'cp' , 'ev1' , 'ev2'] | str , date : int , n_year = 1):
    if denominator == 'mv':
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
        v = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'total_mv' , prev=False , pivot=True)
    elif denominator == 'cp':
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
        v = DATAVENDOR.TRADE.get_trd(date , ['secid','close']).set_index(['secid'])['close'].sort_index()
    elif denominator == 'ev1':
        v = get_ev1_hist(date , n_year)
    elif denominator == 'ev2':
        v = get_ev2_hist(date , n_year)
    else:
        raise KeyError(denominator)
    return v

def valuation_latest(numerator : str , denominator , date : int , fin_type : Literal['qtr' , 'ttm' , 'acc'] , qtr_method = 'diff'):
    kwargs = {'qtr_method' : qtr_method} if numerator.startswith('indi') else {}
    num = get_numerator(f'{numerator}@{fin_type}' , date , **kwargs)
    den = get_denominator(denominator , date)
    return calc_valuation(num , den , pct = False)

def valuation_rank1y(numerator : str , denominator , date : int , fin_type : Literal['qtr' , 'ttm' , 'acc'] , qtr_method = 'diff'):
    kwargs = {'qtr_method' : qtr_method} if numerator.startswith('indi') else {}
    num = get_numerator_hist(f'{numerator}@{fin_type}' , date , 1 , **kwargs)
    den = get_denominator_hist(denominator , date , 1)
    return calc_valuation(num , den , pct = True)

class btop(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '市净率倒数'
    
    def calc_factor(self, date: int):
        pb = DATAVENDOR.TRADE.get_val(date).set_index(['secid'])['pb']
        return calc_valuation(1 , pb , pct = False)
    
class btop_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '市净率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
        pb = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'pb' , prev=False , pivot=True)
        return calc_valuation(1 , pb , pct = True)
    
class dtop(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '滚动分红率'
    
    def calc_factor(self, date: int):
        dv_ttm = DATAVENDOR.TRADE.get_val(date).set_index(['secid'])['dv_ttm'] / 100
        return calc_valuation(dv_ttm , 1 , pct = False)
    
class dtop_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '股东分红率,1年分位数'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'y')
        dv_ttm = DATAVENDOR.TRADE.get_val_data(start_date , end_date , 'dv_ttm' , prev=False , pivot=True)
        return calc_valuation(dv_ttm , 1 , pct = True)

class ebit_ev1_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV(不剔除货币资金)'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@ebit' , 'ev1' , date , 'ttm')

class ebit_ev1_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV(不剔除货币资金),1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@ebit' , 'ev1' , date , 'ttm')
    
class ebitda_ev1_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV(不剔除货币资金)'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@ebitda' , 'ev1' , date , 'ttm')

class ebitda_ev1_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV(不剔除货币资金),1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@ebitda' , 'ev1' , date , 'ttm')
    
class ebit_ev2_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV(剔除货币资金)'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@ebit' , 'ev2' , date , 'ttm')

class ebit_ev2_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV(剔除货币资金),1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@ebit' , 'ev2' , date , 'ttm')

class ebitda_ev2_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV(剔除货币资金)'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@ebitda' , 'ev2' , date , 'ttm')

class ebitda_ev2_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV(剔除货币资金),1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@ebitda' , 'ev2' , date , 'ttm')
    
class etop_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('is@n_income_attr_p' , 'mv' , date , 'ttm')
    
class etop_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('is@n_income_attr_p' , 'mv' , date , 'ttm')
    
class etop_dedt_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM扣非市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@profit_dedt' , 'mv' , date , 'ttm')
    
class etop_dedt_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM扣非市盈率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@profit_dedt' , 'mv' , date , 'ttm')

class etop_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市盈率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('is@n_income_attr_p' , 'mv' , date , 'qtr')

class etop_rank1y_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市盈率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('is@n_income_attr_p' , 'mv' , date , 'qtr')
    
class fcfetop_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM企业股权自由现金流量/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@fcfe' , 'mv' , date , 'ttm')

class fcfetop_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM企业股权自由现金流量/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@fcfe' , 'mv' , date , 'ttm')

class fcfetop_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度企业股权自由现金流量/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('indi@fcfe' , 'mv' , date , 'qtr')

class fcfetop_rank1y_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度企业股权自由现金流量/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('indi@fcfe' , 'mv' , date , 'qtr')
    
class ocftop_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营现金流/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('cf@n_cashflow_act' , 'mv' , date , 'ttm')

class ocftop_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM经营现金流/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('cf@n_cashflow_act' , 'mv' , date , 'ttm')

class ocftop_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营现金流/市值'
    
    def calc_factor(self, date: int):
        return valuation_latest('cf@n_cashflow_act' , 'mv' , date , 'qtr')

class ocftop_rank1y_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度经营现金流/市值,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('cf@n_cashflow_act' , 'mv' , date , 'qtr')
    
class stop_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市销率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('is@revenue' , 'mv' , date , 'ttm')
    
class stop_rank1y_ttm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'TTM市销率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('is@revenue' , 'mv' , date , 'ttm')

class stop_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市销率倒数'
    
    def calc_factor(self, date: int):
        return valuation_latest('is@revenue' , 'mv' , date , 'qtr')

class stop_rank1y_qtr(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '单季度市销率倒数,1年分位数'
    
    def calc_factor(self, date: int):
        return valuation_rank1y('is@revenue' , 'mv' , date , 'qtr')