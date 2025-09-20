from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor


def mdr(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    returns = DATAVENDOR.TRADE.get_returns(start_date , end_date)
    return returns.max()

class mom_mdr1m(MomentumFactor):
    init_date = 20110101
    description = '1个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 1)

class mom_mdr2m(MomentumFactor):
    init_date = 20110101
    description = '2个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 2)
    
class mom_mdr3m(MomentumFactor):
    init_date = 20110101
    description = '3个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 3)
    
class mom_mdr6m(MomentumFactor):
    init_date = 20110101
    description = '6个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 6)
    
class mom_mdr12m(MomentumFactor):
    init_date = 20110101
    description = '12个月区间最大日收益率'

    def calc_factor(self , date : int):
        return mdr(date , 12)