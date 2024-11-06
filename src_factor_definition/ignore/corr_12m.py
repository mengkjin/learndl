import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class corr_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'correlation'
    description = '12个月相关系数'
    
    def calc_factor(self, date: int):
        """计算12个月相关系数
        1. 获取过去12个月的个股和市场收益率数据
        2. 计算相关系数
        """
        # TODO: 需要定义获取收益率数据的接口
        start_date = TradeDate(date) - 240  # 约12个月的交易日
        stock_returns = TRADE_DATA.get_returns(start_date, date)
        market_returns = TRADE_DATA.get_market_returns(start_date, date)
        
        # 计算相关系数
        def calc_corr(x): 
            return pd.Series(x).corr(market_returns)
            
        corr = stock_returns.apply(calc_corr)
        return corr.rename('factor_value').to_frame() 