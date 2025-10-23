import pandas as pd

from src.data import DATAVENDOR
from src.basic import DB
from src.res.factor.calculator import RegimeFactor

__all__ = [
    'platform_breakout'
]

class platform_breakout(RegimeFactor):
    init_date = 20110101
    description = '横盘突破,过去交易日通道宽度收缩环境下,当日大涨'

    @classmethod
    def _platform_breakout(cls , start : int | None = None , end : int | None = None) -> pd.DataFrame:
        quote = DB.load('index_daily_ts' , '000985.CSI')
        if start is not None:
            quote = quote.query('trade_date >= @start')
        if end is not None:
            quote = quote.query('trade_date <= @end')

        quote = quote.sort_values('trade_date').copy()
        quote['bandwidth'] = quote['high'].rolling(5).max() - quote['low'].rolling(5).min()
        quote['narrowing_movement'] = quote['bandwidth'].shift(1) < quote['bandwidth'].shift(2) # 过去2天通道宽度收缩
        quote['platform'] = quote['pct_chg'].abs().shift(1).rolling(5).max() < 1. # 过去5天涨跌幅均小于1%
        quote['breakout'] = quote['pct_chg'] > 1. # 当日大涨    
        quote[cls.factor_name] = quote['narrowing_movement'] & quote['platform'] & quote['breakout']

        quote = quote.rename(columns = {'trade_date' : 'date'}).\
            filter(items = ['date' , cls.factor_name , 'narrowing_movement' , 'platform' , 'breakout']).\
            reset_index(drop = True)
        return quote

    def calc_history(self , date : int) -> pd.DataFrame:
        df = self._platform_breakout(end = date)
        return df
    
    def calc_factor(self, date: int) -> pd.DataFrame:
        max_date = self.max_date()
        start_date = DATAVENDOR.CALENDAR.td(max_date , -7)
        df = self._platform_breakout(start_date , date).query('date > @max_date & date <= @date')
        return df