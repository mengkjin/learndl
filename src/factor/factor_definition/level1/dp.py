import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class dp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '股息率'
    
    def calc_factor(self, date: int):
        """计算股息率
        1. 获取股息和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        dividend = TRADE_DATA.get_dividend(date)
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = dividend / market_value
        return ratio.rename('factor_value').to_frame() 