import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class corr_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'correlation'
    description = '相关系数分位数'
    
    def calc_factor(self, date: int):
        """计算相关系数的1年分位数
        1. 获取过去1年的相关系数数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取收益率数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        stock_returns = TRADE_DATA.get_returns(start_date, date)
        market_returns = TRADE_DATA.get_market_returns(start_date, date)
        
        # 计算滚动相关系数
        def calc_rolling_corr(x):
            return pd.Series(x).rolling(20).corr(market_returns)
            
        corr = stock_returns.apply(calc_rolling_corr)
        rank = corr.rank(pct=True)
        return rank.rename('factor_value').to_frame() 