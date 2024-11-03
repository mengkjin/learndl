import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class beta_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'volatility'
    description = '3个月贝塔'
    
    def calc_factor(self, date: int):
        """计算3个月贝塔
        1. 获取过去3个月的个股和市场收益率数据
        2. 计算贝塔系数
        """
        # TODO: 需要定义获取收益率数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        stock_returns = TRADE_DATA.get_returns(start_date, date)
        market_returns = TRADE_DATA.get_market_returns(start_date, date)
        
        # 计算贝塔系数
        def calc_beta(x): 
            return pd.Series(x).cov(market_returns) / market_returns.var()
            
        beta = stock_returns.apply(calc_beta)
        return beta.rename('factor_value').to_frame() 