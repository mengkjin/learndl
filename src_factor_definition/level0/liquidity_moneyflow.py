import numpy as np
import pandas as pd
from typing import Any , Literal

from src.factor.classes import StockFactorCalculator
from src.data import TSData

def inflow_by_return(date , n_months : int , direction : Literal[-1,1] , div_amt = True , min_finite_ratio = 0.25):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm')
    net_mfs   = TSData.TRADE.get_mf_data(start_date , end_date , 'net_mf_amount' , pivot = True)
    rets_sign = np.sign(TSData.TRADE.get_returns(start_date , end_date , pivot = True))

    net_mfs *= 1 * (rets_sign == direction) + 0.5 * (rets_sign == 0)
    inflow = net_mfs.sum()
    inflow[np.isfinite(net_mfs).sum() < len(net_mfs) * min_finite_ratio] = np.nan
    if div_amt:
        inflow /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True).sum()
    return inflow

def moneyflow_type(date , n_months : int , size : Literal['sm' , 'md' , 'lg' , 'elg'] , 
                   act : Literal['buy' , 'sell'] , vol_type : Literal['volume' , 'amount'] , div_amt = False):
    start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm')
    mf_type : Any = f'{act}_{size}_' + 'vol' if vol_type == 'volume' else 'amount'
    mf   = TSData.TRADE.get_mf_data(start_date , end_date , mf_type , pivot = True)
    if div_amt:
        mf /= TSData.TRADE.get_volumes(start_date , end_date , vol_type , pivot = True)
    return mf
    
class bsact_neg_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流入-负收益'
    
    def calc_factor(self, date: int):
        return inflow_by_return(date , 1 , -1)
    
class bsact_pos_1m(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流入-正收益'
    
    def calc_factor(self, date: int):
        return inflow_by_return(date , 1 , 1)
    
class flow_small(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流强度--小单'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_sm_amount','sell_sm_amount'] , pivot = True)
        net_mf = mf['buy_sm_amount'] - mf['buy_sm_amount']
        net_mf /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf
    
class flow_medium(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流强度--中单'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_md_amount','sell_md_amount'] , pivot = True)
        net_mf = mf['buy_md_amount'] - mf['sell_md_amount']
        net_mf /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf
    
class flow_large(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流强度--大单'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_lg_amount','sell_lg_amount'] , pivot = True)
        net_mf = mf['buy_lg_amount'] - mf['sell_lg_amount']
        net_mf /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf
    
class flow_exlarge(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月资金流强度--超大单'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_elg_amount'] , pivot = True)
        net_mf = mf['buy_elg_amount'] - mf['sell_elg_amount']
        net_mf /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf
    
class flow_elsm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月超大单-小单占比'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_sm_amount'] , pivot = True)
        net_mf = mf['buy_elg_amount'] - mf['sell_sm_amount']
        net_mf /= TSData.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf

class flow_corr_elsm(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月超大单-小单相关性'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 1 , 'm')
        mf = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_elg_amount','buy_sm_amount','sell_sm_amount'] , pivot = True)
        el = mf['buy_elg_amount'] - mf['sell_elg_amount']
        sm = mf['buy_sm_amount'] - mf['buy_sm_amount']

        corr = el.corr(sm)
        return corr

class flow_corr_smlag(StockFactorCalculator):
    init_date = 20110101
    category0 = 'behavior'
    category1 = 'liquidity'
    description = '1个月小单滞后相关性'
    
    def calc_factor(self, date: int):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , 22 , 'd')
        mf = TSData.TRADE.get_mf_data(start_date , end_date , ['buy_sm_amount','sell_sm_amount'] , pivot = True)
        sm = mf['buy_sm_amount'] - mf['buy_sm_amount']

        corr = sm.corr(sm.shift(1))
        return corr
