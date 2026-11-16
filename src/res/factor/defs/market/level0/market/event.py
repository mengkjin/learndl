"""
Market event factors for the project
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from src.proj import DB
from src.data import DATAVENDOR
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
        from src.proj import CALENDAR
        codes = DB.load('information_ts', 'zx_industry')['l1_code'].dropna().unique()
        quote = DB.load('index_daily_ts', '000985.CSI')
        if quote.empty or not len(codes):
            raise ValueError('high_level_switch: missing index/industry master')
        end = int(end if end is not None else quote.trade_date.max())
        dates = np.asarray(CALENDAR.range(int(quote.trade_date.min()), end, 'td'))
        industries = {code: DB.load('index_daily_ts', code) for code in codes}
        result = cls.from_quotes(quote, industries, dates)
        return result if start is None else result.loc[result.date >= start].reset_index(drop=True)

    @staticmethod
    def from_quotes(quote, industries, dates):
        """Compute on full prehistory; missing source days never become false events."""
        def align(frame, required):
            if frame.empty:
                return pd.DataFrame(index=dates, columns=required, dtype=float)
            if not {'trade_date', *required} <= set(frame):
                raise ValueError('Malformed market event source')
            frame = frame.drop_duplicates().copy()
            if frame.trade_date.isna().any() or frame.trade_date.duplicated().any():
                raise ValueError('Conflicting/invalid market event dates')
            return frame.set_index('trade_date').sort_index().reindex(dates)[required].replace([np.inf, -np.inf], np.nan)
        q = align(quote, ['high', 'low', 'close', 'pct_chg'])
        peaked, peaking, required, available = [], [], [], []
        for frame in industries.values():
            c = align(frame, ['close']).close
            # Historical industry membership is unavailable. An industry enters the
            # required universe after its first 244 calendar observations; this is causal.
            first = np.flatnonzero(c.notna().to_numpy())
            active = pd.Series(False, index=dates)
            if len(first):
                active.iloc[int(first[0]) + 243:] = True
            else:
                active[:] = True  # missing entire industry source is not zero contribution
            high = c.rolling(243, min_periods=243).max()
            valid = c.notna() & c.shift(1).notna() & high.shift(1).notna()
            peaked.append((c.shift(1) >= high.shift(1)).astype(float).where(valid))
            peaking.append((c >= high.shift(1)).astype(float).where(valid))
            required.append(active)
            available.append(valid & active)
        if not peaked:
            raise ValueError('No industry sources')
        req = pd.concat(required, axis=1).sum(axis=1)
        count = pd.concat(available, axis=1).sum(axis=1)
        q['peaked'] = pd.concat(peaked, axis=1).sum(axis=1, min_count=1)
        q['peaking'] = pd.concat(peaking, axis=1).sum(axis=1, min_count=1)
        prev = q.close.shift(1)
        parts = pd.concat([q.high-q.low, (q.high-prev).abs(), (q.low-prev).abs()], axis=1)
        q['tr'] = (parts.max(axis=1) / prev).where(parts.notna().all(axis=1) & (prev > 0))
        q['atr'] = q.tr.rolling(60, min_periods=60).mean()
        market_valid = q.atr.notna() & q.pct_chg.notna()
        q['market_selloff'] = (q.pct_chg / 100 < -q.atr).astype(float).where(market_valid)
        valid = market_valid & (req > 0) & (count == req)
        q['high_level_switch'] = ((q.peaked-q.peaking >= 3) & (q.market_selloff == 1)).astype(float).where(valid)
        q['source_valid'] = valid.astype(float)
        q['industry_required'] = req
        q['industry_available'] = count
        q['source_reason'] = np.where(~market_valid, 'market_missing_or_warmup',
                                     np.where(req == 0, 'industry_warmup',
                                              np.where(count != req, 'industry_missing', 'valid')))
        return q.rename_axis('date').reset_index()[['date', 'high_level_switch', 'peaked', 'peaking',
                    'market_selloff', 'tr', 'atr', 'source_valid', 'source_reason',
                    'industry_required', 'industry_available']]

    def validate_value(self, df, date, strict=False):
        valid = df.source_valid.eq(1)
        if not df.source_valid.isin([0, 1]).all() or not df.loc[valid, self.factor_name].isin([0, 1]).all():
            raise ValueError('Invalid high_level_switch source mask/label')
        if df.loc[~valid, self.factor_name].notna().any():
            raise ValueError('Unknown event must remain NaN')
        if df.date.duplicated().any():
            raise ValueError('Duplicate event dates')
        return df

    def calc_history(self , date : int) -> pd.DataFrame:
        df = self._high_level_switch(end = date)
        return df
    
    def calc_factor(self, date: int) -> pd.DataFrame:
        max_date = min(self.max_date , date)
        start = DATAVENDOR.CALENDAR.td(max_date , -250).as_int()
        df = self._high_level_switch(start , date).query('date <= @date')
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

    singal_columns = ['selloff' , 'rebound' , 'selloff_start' , 'rebound_start' , 'minimum_rebound' , 'minimum_selloff' , 'tr' , 'atr' , 'parameter_magnifier']

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
        quote['selloff_start'] = -1
        quote['rebound_start'] = -1
        selloff_level = 1.
        rebound_level = 1.
        selloff_start = -1
        rebound_start = -1
        for row , q in quote.iterrows():
            assert isinstance(row , int)
            if row == 0:
                prev_selloff = 0.
                prev_selloff_start = int(q['trade_date'])
            else:
                prev_selloff = quote['selloff'][row - 1]
                prev_selloff_start = quote['selloff_start'][row - 1]

            if q['pct_chg'] > 0:
                rebound_level = rebound_level * (1 + q['pct_chg'] / 100)
                quote.loc[row , 'rebound'] = rebound_level - 1
                if rebound_start == -1:
                    rebound_start = int(q['trade_date'])
                quote.loc[row , 'rebound_start'] = rebound_start
            else:
                rebound_level = 1.
                rebound_start = -1

            if rebound_level > 1 + q['minimum_rebound']:
                selloff_level = 1.
                selloff_start = -1
                quote.loc[row , 'selloff'] = prev_selloff
                quote.loc[row , 'selloff_start'] = prev_selloff_start
            else:
                selloff_level = selloff_level * (1 + q['pct_chg'] / 100)
                if selloff_start == -1:
                    selloff_start = int(q['trade_date'])
                    quote.loc[row , 'selloff'] = selloff_level - 1
                    quote.loc[row , 'selloff_start'] = selloff_start
                elif row > 0:
                    quote.loc[row , 'selloff'] = min(prev_selloff , selloff_level - 1)
                    quote.loc[row , 'selloff_start'] = prev_selloff_start
                    
        quote['selloff_rebound'] = 1 * (quote['selloff'] < quote['minimum_selloff']) * (quote['rebound'] > quote['minimum_rebound'])
        quote['selloff_rebound'] = quote['selloff_rebound'].where(quote['selloff_rebound'].shift(1) != 1 , 0)
        quote = quote.rename(columns = {'trade_date' : 'date'}).filter(items = ['date' , cls.factor_name , *cls.singal_columns]).reset_index(drop = True)
        return quote

    def calc_history(self , date : int) -> pd.DataFrame:
        df = self._selloff_rebound(end = date)
        return df
    
    def calc_factor(self, date: int) -> pd.DataFrame:
        max_date = min(self.max_date , date)
        start = DATAVENDOR.CALENDAR.td(max_date , -250).as_int()
        df = self._selloff_rebound(start , date).query('date <= @date')
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
        max_date = min(self.max_date , date)
        start = DATAVENDOR.CALENDAR.td(max_date , -7).as_int()
        df = self._platform_breakout(start , date).query('date <= @date')
        return df