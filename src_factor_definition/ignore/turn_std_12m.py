import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class turn_std_12m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'volume'
    description = '12个月换手率波动率'
    
    def calc_factor(self, date: int):
        """计算12个月换手率波动率
        1. 获取过去12个月的换手率数据
        2. 计算标准差
        """
        # TODO: 需要定义获取换手率数据的接口
        start_date = TradeDate(date) - 240  # 约12个月的交易日
        turnover = TRADE_DATA.get_turnover(start_date, date)
        
        std = turnover.std()
        return std.rename('factor_value').to_frame() 