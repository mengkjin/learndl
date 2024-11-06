import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class amount_bias_1m(StockFactorCalculator):
    init_date = 20070101
    category0 = 'market'
    category1 = 'sentiment'
    description = '1个月成交额乖离率'
    
    def calc_factor(self, date: int):
        """计算1个月成交额乖离率
        1. 获取过去1个月的成交额数据
        2. 计算当前值与均值的偏离程度
        """
        # TODO: 需要定义获取成交额数据的接口
        start_date = TradeDate(date) - 20  # 约1个月的交易日
        amount = TRADE_DATA.get_amount(start_date, date)
        
        # 计算乖离率
        current = amount.iloc[-1]
        mean = amount.mean()
        bias = (current - mean) / mean
        
        return bias.rename('factor_value').to_frame() 