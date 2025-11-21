from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor


def phigh(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    high = DATAVENDOR.TRADE.get_quotes(start_date,end_date,'high',pivot=True).max()
    cp   = DATAVENDOR.TRADE.get_quotes(end_date,end_date,'close',pivot=True).iloc[-1]
    mom  = cp / high - 1
    return mom

class mom_phigh1m(MomentumFactor):
    init_date = 20110101
    description = '1个月最高价距离'

    def calc_factor(self , date : int):
        return phigh(date , 1)