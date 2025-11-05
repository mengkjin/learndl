import pandas as pd

from src.data import DATAVENDOR
from src.basic import DB
from src.res.factor.calculator import MarketEventFactor

__all__ = [
    'high_level_switch' ,
    'selloff_rebound' ,
    'platform_breakout'
]

def slice_quote(df : pd.DataFrame , start : int | None = None , end : int | None = None) -> pd.DataFrame:
    df['trade_date'] = df['trade_date'].astype(int)
    if start is not None:
        df = df.query('trade_date >= @start')
    if end is not None:
        df = df.query('trade_date <= @end')
    df = df.sort_values('trade_date').reset_index(drop = True)
    return df

class high_level_switch(MarketEventFactor):
    """
    高位出现切换
    1. 中信1级行业至少3个前一天创52周新高但今日至少少了3个
    2. 市场下跌幅度超过ATR60
    """
    init_date = 20110101
    description = '高位出现切换'

    singal_columns = ['peaked' , 'peaking' , 'market_selloff' , 'tr' , 'atr']

    @classmethod
    def _high_level_switch(cls , start : int | None = None , end : int | None = None) -> pd.DataFrame:
        level1_codes = DB.load('information_ts' , 'zx_industry')['l1_code'].unique()

        index_quotes = {
            ts_code : DB.load('index_daily_ts' , ts_code).reset_index(drop = True) for ts_code in level1_codes
        }

        for index , df in index_quotes.items():
            df = slice_quote(df , start , end).copy()
            df['52week_high'] = df['close'].rolling(243).max()
            df['peaked'] = 1 * (df['close'] >= df['52week_high']).shift(1)
            df['peaking'] = 1 * (df['close'] >= df['52week_high'].shift(1))
            index_quotes[index] = df.filter(items = ['trade_date' , 'peaked' , 'peaking'])

        l1_index_stat = pd.concat(index_quotes.values()).groupby('trade_date').sum().reset_index()

        quote = DB.load('index_daily_ts' , '000985.CSI')
        quote = slice_quote(quote , start , end)

        quote['tr'] = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['close'].shift(1)).abs() , (quote['low'] - quote['close'].shift(1)).abs()] , axis = 1).max(axis = 1) / quote['close'].shift(1)
        quote['atr'] = quote['tr'].rolling(60).mean()
        quote['market_selloff'] = 1 * (quote['pct_chg'] / 100 < -quote['atr'])

        quote = quote.merge(l1_index_stat , on = 'trade_date')

        quote['high_level_switch'] = 1 * (quote['peaked'] - quote['peaking'] >= 3) * (quote['market_selloff'] == 1)
        quote = quote.rename(columns = {'trade_date' : 'date'}).filter(items = ['date' , cls.factor_name , *cls.singal_columns]).reset_index(drop = True)
        return quote

    def calc_history(self , date : int) -> pd.DataFrame:
        df = self._high_level_switch(end = date)
        return df
    
    def calc_factor(self, date: int) -> pd.DataFrame:
        max_date = min(self.max_date() , date)
        start_date = DATAVENDOR.CALENDAR.td(max_date , -250)
        df = self._high_level_switch(start_date , date).query('date <= @date')
        return df

class selloff_rebound(MarketEventFactor):
    """
    暴跌反弹,过去交易日大跌环境下,当日反弹
    1. 过去下跌幅度超过5%
    2. 最新反弹幅度超过0.5%,且此前在下跌途中没有反弹超过0.5%
    3. 基于过去60天TR均值,放大和缩小反弹与下跌幅度的阈值
    """
    init_date = 20110101
    description = '暴跌反弹,过去交易日大跌环境下,当日反弹'

    singal_columns = ['trigger_rebound' , 'selloff' , 'rebound' , 'minimum_rebound' , 'minimum_selloff' , 'tr' , 'atr' , 'parameter_magnifier']

    @classmethod
    def _selloff_rebound(cls , start : int | None = None , end : int | None = None) -> pd.DataFrame:
        quote = DB.load('index_daily_ts' , '000985.CSI')
        quote = slice_quote(quote , start , end)

        quote['tr'] = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['close'].shift(1)).abs() , (quote['low'] - quote['close'].shift(1)).abs()] , axis = 1).max(axis = 1) / quote['close'].shift(1)
        quote['atr'] = quote['tr'].rolling(60).mean()

        quote['parameter_magnifier'] = 1.
        quote['parameter_magnifier'] *= (quote['atr'] / 0.01).pow(0.5).where(quote['atr'] < 0.01 , 1.)
        quote['parameter_magnifier'] *= (quote['atr'] / 0.02).pow(0.5).where(quote['atr'] > 0.02 , 1.)

        quote['minimum_rebound'] = 0.005 * quote['parameter_magnifier']
        quote['minimum_selloff'] = -0.05 * quote['parameter_magnifier']

        quote['selloff'] = 0.
        quote['rebound'] = 0.
        cr = 1.
        so = 1.
        reset_selloff = False
        for row , q in quote.iterrows():
            assert isinstance(row , int)
            prev_selloff = quote['selloff'][row - 1] if row > 0 else 0.
            if q['pct_chg'] > 0:
                cr = cr * (1 + q['pct_chg'] / 100)
                quote.loc[row , 'rebound'] = cr - 1
            else:
                cr = 1.
            if cr > 1 + q['minimum_rebound']:
                so = 1.
                quote.loc[row , 'selloff'] = prev_selloff
                reset_selloff = True
            else:
                so = so * (1 + q['pct_chg'] / 100)
                if reset_selloff:
                    quote.loc[row , 'selloff'] = so - 1
                elif row > 0:
                    quote.loc[row , 'selloff'] = min(prev_selloff , so - 1)
                reset_selloff = False

        quote['selloff_rebound'] = 1 * (quote['selloff'] < quote['minimum_selloff']) * (quote['rebound'] > quote['minimum_rebound'])
        quote['trigger_rebound'] = quote['selloff_rebound'].where(quote['selloff_rebound'].shift(1) != 1 , 0)

        quote = quote.rename(columns = {'trade_date' : 'date'}).filter(items = ['date' , cls.factor_name , *cls.singal_columns]).reset_index(drop = True)
        return quote

    def calc_history(self , date : int) -> pd.DataFrame:
        df = self._selloff_rebound(end = date)
        return df
    
    def calc_factor(self, date: int) -> pd.DataFrame:
        max_date = min(self.max_date() , date)
        start_date = DATAVENDOR.CALENDAR.td(max_date , -250)
        df = self._selloff_rebound(start_date , date).query('date <= @date')
        return df

class platform_breakout(MarketEventFactor):
    """
    横盘突破,过去交易日通道宽度收缩环境下,当日大涨
    1. 过去5天涨跌幅均小于1%
    2. 过去2天通道宽度收缩
    3. 当日大涨超过1%
    """
    init_date = 20110101
    description = '横盘突破,过去交易日通道宽度收缩环境下,当日大涨'

    @classmethod
    def _platform_breakout(cls , start : int | None = None , end : int | None = None) -> pd.DataFrame:
        quote = DB.load('index_daily_ts' , '000985.CSI')
        quote = slice_quote(quote , start , end)

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
        max_date = min(self.max_date() , date)
        start_date = DATAVENDOR.CALENDAR.td(max_date , -7)
        df = self._platform_breakout(start_date , date).query('date <= @date')
        return df