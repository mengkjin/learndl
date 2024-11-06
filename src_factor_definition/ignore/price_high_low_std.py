import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class price_high_low_std(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'others'
    description = '最高最低价比值波动率'
    
    def calc_factor(self, date: int):
        """计算最高最低价比值的波动率
        1. 获取过去一段时间的最高价和最低价数据
        2. 计算每日比值的标准差
        """
        # TODO: 需要定义获取价格数据的接口
        start_date = TradeDate(date) - 20  # 约1个月的交易日
        high_prices = TRADE_DATA.get_high_price(start_date, date)
        low_prices = TRADE_DATA.get_low_price(start_date, date)
        
        ratio = high_prices / low_prices
        std = ratio.std()
        
        return std.rename('factor_value').to_frame() 