import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data import TSData
from src.func.transform import time_weight , apply_ols
from src.func.singleton import singleton_threadsafe

class _FamaFrench3:
    N_MONTHS : int = 1
    def __init__(self , date):
        if getattr(self, 'date') is None or self.date != int(date):
            self.date = int(date)
            self.regression(self.date , self.N_MONTHS)

    def regression(self , date , n_months : int , half_life = 0 , min_finite_ratio = 0.25):
        start_date , end_date = TSData.CALENDAR.td_start_end(date , n_months , 'm')
        rets = TSData.TRADE.get_returns(start_date , end_date , mask = False , pivot = False)
        circ = TSData.TRADE.get_mv(start_date , end_date , mv_type = 'circ_mv' , pivot = False)
        rets = rets.merge(circ , on = ['date' , 'secid'])
        rets['mv_change'] = rets['pctchange'] * rets['circ_mv']

        stk_ret = TSData.TRADE.get_returns(start_date, end_date)
        mkt_ret = TSData.TRADE.get_market_return(start_date, end_date)
        wgt = time_weight(len(mkt_ret) , half_life)
        b = apply_ols(mkt_ret.values.flatten() , stk_ret.values , wgt)[1]
        b[np.isfinite(stk_ret).sum() < len(mkt_ret) * min_finite_ratio] = np.nan
        return pd.Series(b , index = stk_ret.columns)
    
@singleton_threadsafe
class FF3_1m(_FamaFrench3):
    N_MONTHS = 1

@singleton_threadsafe
class FF3_2m(_FamaFrench3):
    N_MONTHS = 2

@singleton_threadsafe
class FF3_3m(_FamaFrench3):
    N_MONTHS = 3    

@singleton_threadsafe
class FF3_6m(_FamaFrench3):
    N_MONTHS = 6

@singleton_threadsafe
class FF3_12m(_FamaFrench3):
    N_MONTHS = 12