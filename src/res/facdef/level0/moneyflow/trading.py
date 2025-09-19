import numpy as np
from typing import Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'bsact_neg_1m' , 'bsact_pos_1m' , 'flow_small' , 'flow_medium' , 'flow_large' , 'flow_exlarge' ,
    'flow_elsm' , 'flow_corr_elsm' , 'flow_corr_smlag'
]

def inflow_by_return(date , n_months : int , direction : Literal[-1,1] , div_amt = True , min_finite_ratio = 0.25):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm')
    net_mfs   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , 'net_mf_amount' , pivot = True)
    rets_sign = np.sign(DATAVENDOR.TRADE.get_returns(start_date , end_date , pivot = True))

    net_mfs *= 1 * (rets_sign == direction) + 0.5 * (rets_sign == 0)
    inflow = net_mfs.sum()
    inflow[np.isfinite(net_mfs).sum() < len(net_mfs) * min_finite_ratio] = np.nan
    if div_amt:
        inflow /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True).sum()
    return inflow
    
class bsact_neg_1m(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流入-负收益'
    
    def calc_factor(self, date: int):
        return inflow_by_return(date , 1 , -1)
    
class bsact_pos_1m(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流入-正收益'
    
    def calc_factor(self, date: int):
        return inflow_by_return(date , 1 , 1)
    
class flow_small(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流强度--小单'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_sm_amount','sell_sm_amount'] , pivot = True)
        net_mf = mf['buy_sm_amount'] - mf['sell_sm_amount']
        net_mf /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf.mean()
    
class flow_medium(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流强度--中单'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_md_amount','sell_md_amount'] , pivot = True)
        net_mf = mf['buy_md_amount'] - mf['sell_md_amount']
        net_mf /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf.mean()
    
class flow_large(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流强度--大单'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_lg_amount','sell_lg_amount'] , pivot = True)
        net_mf = mf['buy_lg_amount'] - mf['sell_lg_amount']
        net_mf /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf.mean()
    
class flow_exlarge(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月资金流强度--超大单'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_elg_amount'] , pivot = True)
        net_mf = mf['buy_elg_amount'] - mf['sell_elg_amount']
        net_mf /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf.mean()
    
class flow_elsm(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月超大单-小单占比'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf   = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_sm_amount'] , pivot = True)
        net_mf = mf['buy_elg_amount'] - mf['sell_sm_amount']
        net_mf /= DATAVENDOR.TRADE.get_volumes(start_date , end_date , 'amount' , pivot = True)
        return net_mf.mean()

class flow_corr_elsm(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月超大单-小单相关性'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 1 , 'm')
        mf = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_elg_amount','sell_elg_amount','buy_sm_amount','sell_sm_amount'] , pivot = True)
        el = mf['buy_elg_amount'] - mf['sell_elg_amount']
        sm = mf['buy_sm_amount'] - mf['sell_sm_amount']

        corr = el.corrwith(sm)
        return corr
    
class flow_corr_smlag(StockFactorCalculator):
    init_date = 20110101
    category1 = 'trading'
    description = '1个月小单滞后相关性'
    
    def calc_factor(self, date: int):
        start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , 22 , 'd')
        mf = DATAVENDOR.TRADE.get_mf_data(start_date , end_date , ['buy_sm_amount','sell_sm_amount'] , pivot = True)
        sm = mf['buy_sm_amount'] - mf['sell_sm_amount']

        corr = sm.corrwith(sm.shift(1))
        return corr