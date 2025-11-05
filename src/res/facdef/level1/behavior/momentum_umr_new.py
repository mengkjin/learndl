import pandas as pd
import numpy as np

from src.basic import CALENDAR , DB , CONF
from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor

from src.func.transform import time_weight , lm_resid
from src.func.linalg import symmetric_orth_np

_cached_data = {}

def get_market_event_dates():
    if 'market_event_dates' not in _cached_data:
        market_events = [
            DB.load('market_factor' , 'high_level_switch').query('high_level_switch == 1')['date'].to_numpy() ,
            DB.load('market_factor' , 'platform_breakout').query('platform_breakout == 1')['date'].to_numpy() ,
            DB.load('market_factor' , 'selloff_rebound').query('trigger_rebound == 1')['date'].to_numpy()
        ]
        market_event_dates = np.unique(np.concatenate(market_events))
        _cached_data['market_event_dates'] = market_event_dates
    return _cached_data['market_event_dates']  

def umr_new_all(date , n_months : int , risk_window : int = 10):
    risk_type_list = ['true_range' , 'turnover' , 'large_buy_pdev' , 'small_buy_pct' ,
        'sqrt_avg_size' , 'open_close_pct' , 'ret_volatility' , 'ret_skewness']
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm')

    rets = DATAVENDOR.TRADE.get_returns(start_date , end_date , mask = True)
    mkt_ret = DATAVENDOR.TRADE.get_market_return(start_date , end_date)
    exc_rets = rets - mkt_ret.values

    n_days = exc_rets.shape[0]
    p_neg_rets = ((exc_rets < 0).sum(axis = 0) / n_days).rename('p_neg_rets')

    wgt = time_weight(n_days , int(n_days / 2)).reshape(-1,1)

    trailing_qe = CALENDAR.qe_trailing(date , 5)
    disclosure_dates = DATAVENDOR.FINA.gets(trailing_qe , 'disclosure').reset_index(drop = True).\
        filter(items = ['secid' , 'actual_date']).sort_values(['secid' , 'actual_date']).\
            reset_index(drop = True).rename(columns = {'actual_date' : 'date'}).dropna()
    disclosure_dates['date'] = disclosure_dates['date'].astype(int)

    market_event_dates = get_market_event_dates()

    mv_indus = DATAVENDOR.RISK.get_exp(date , ['secid' , 'size'] + CONF.Factor.RISK.indus).\
        reset_index(drop = True).set_index('secid').reindex(rets.columns)
    commom_x = mv_indus.join(p_neg_rets)

    risk_start_date = max(DATAVENDOR.CALENDAR.td(start_date , -2 * risk_window + 1) , DB.min_date('exposure' , 'daily_risk'))
    umrs : dict[str , pd.Series] = {}
    for risk_type in risk_type_list:
        risks = DATAVENDOR.EXPO.get_risks(risk_start_date , end_date , field = risk_type , pivot = True)
        special_dates = disclosure_dates.query('date > @risk_start_date & date <= @end_date').\
            assign(value = 1).pivot_table('value' , 'date' , 'secid').reindex_like(risks)
        special_dates.loc[special_dates.index.isin(market_event_dates)] = 1.

        _min_risk = risks.rolling(risk_window).min()
        risks = risks.where(special_dates.isna() , _min_risk)

        _avg_risk = risks.rolling(risk_window).mean().tail(n_days)
        exc_risk = _avg_risk - risks.tail(n_days)
        exc_risk = exc_risk.dropna(how = 'all')

        p_neg_risk = ((exc_risk < 0).sum(axis = 0) / n_days).rename('p_neg_risk')
        x = commom_x.join(p_neg_risk)

        umr = (exc_rets.tail(exc_risk.shape[0]) * wgt[-exc_risk.shape[0]:] * exc_risk).sum(axis = 0).reindex(rets.columns)

        umr_resid = lm_resid(umr , x , normalize = True)
        umrs[risk_type] = umr_resid
    
    all_umr = pd.concat(umrs.values() , axis = 1).fillna(0)
    orth_umr = symmetric_orth_np(all_umr.to_numpy() , standardize = False).mean(axis = 1)
    
    df = pd.Series({'umr_new': lm_resid(orth_umr , None , normalize = True)} , index = all_umr.index)
    return df

class umr_new_1m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '1个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_new_all(date , 1)

class umr_new_3m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '3个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_new_all(date , 3)

class umr_new_6m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '6个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_new_all(date , 6)

class umr_new_12m(MomentumFactor):
    init_date = 20110101
    update_step = 1
    description = '12个月统一反转因子,原始计算'
    
    def calc_factor(self, date: int):
        return umr_new_all(date , 12)