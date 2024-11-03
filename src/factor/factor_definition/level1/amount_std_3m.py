import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amount_std_3m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '3个月成交额波动率'
    
    def calc_factor(self, date: int):
        """计算3个月成交额波动率
        1. 获取过去3个月的成交额数据
        2. 计算标准差
        """
        # TODO: 需要定义获取成交额数据的接口
        start_date = TradeDate(date) - 60  # 约3个月的交易日
        amount = TRADE_DATA.get_amount(start_date, date)
        
        std = amount.std()
        return std.rename('factor_value').to_frame() 