from src.data import DATAVENDOR
from src.res.factor.calculator import MomentumFactor


def price_weivol(date , n_months : int , lag_months : int = 0):
    start_date , end_date = DATAVENDOR.CALENDAR.td_start_end(date , n_months , 'm' , lag_months)
    vol  = DATAVENDOR.TRADE.get_quotes(start_date,end_date,'volume',pivot=True)
    cp   = DATAVENDOR.TRADE.get_quotes(start_date,end_date,'close',pivot=True)
    weivol = (vol * cp).sum() / vol.sum() / cp.iloc[-1]
    return weivol

class price_weivol1m(MomentumFactor):
    init_date = 20110101
    description = '1个月成交量加权收盘价比值'

    def calc_factor(self , date : int):
        return price_weivol(date , 1)