import pandas as pd
from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor

from src.func.transform import time_weight

def umr_raw_all(date , n_months : int , risk_window : int = 10):
    risk_type_list = ['true_range' , 'turnover' , 'large_buy_pdev' , 'small_buy_pct' ,
        'sqrt_avg_size' , 'open_close_pct' , 'ret_volatility' , 'ret_skewness']
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm')

    rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , mask = True)
    mkt_ret = DATAVENDOR.TRADE.get_market_return(start_date , end_date)
    exc_rets = rets - mkt_ret.values

    n_days = exc_rets.shape[0]
    wgt = time_weight(n_days , int(n_days / 2)).reshape(-1,1)

    risk_start_date = DATAVENDOR.CALENDAR.td(start_date , -risk_window + 1)
    umrs : dict[str , pd.Series] = {}
    for risk_type in risk_type_list:
        risks = DATAVENDOR.EXPO.get_risks(risk_start_date , end_date , field = risk_type , pivot = True)
        avg_risk = risks.rolling(risk_window).mean().tail(n_days)
        exc_risk = avg_risk - risks.tail(n_days)
        umr = (exc_rets * wgt * exc_risk).sum(axis = 0).reindex(rets.columns)
        umrs[risk_type] = umr
    all_umr = pd.concat(umrs.values() , axis = 1).mean(axis = 1).rename('umr_raw')
    return all_umr

class umr_raw_1m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '1个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_raw_all(date , 1)

class umr_raw_3m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '3个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_raw_all(date , 3)

class umr_raw_6m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '6个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_raw_all(date , 6)

class umr_raw_12m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '12个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_raw_all(date , 12)