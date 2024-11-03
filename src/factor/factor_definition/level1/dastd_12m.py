import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class dastd_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'volatility'
    description = '12个月日波动率'
    
    def calc_factor(self, date: int):
        """计算12个月日波动率
        1. 获取过去12个月的日收益率数据
        2. 计算标准差
        """
        # TODO: 需要定义获取收益率数据的接口
        start_date = TradeDate(date) - 240  # 约12个月的交易日
        returns = TRADE_DATA.get_returns(start_date, date)
        
        std = returns.std()
        return std.rename('factor_value').to_frame() 