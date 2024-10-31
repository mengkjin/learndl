import numpy as np
import pandas as pd

from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR , TRADE_DATA , TradeDate

class ret20_raw(StockFactorCalculator):
    init_date = 20070101
    category0 = 'behavior'
    category1 = 'momentum'
    description = 'raw return of 20 days'

    def calc_factor(self , date : int):
        ret = TRADE_DATA.get_rets(TradeDate(date) - 19 , date)
        df = (np.exp(np.log(ret + 1).sum(axis = 0)) - 1).rename('factor_value').to_frame()
        return df