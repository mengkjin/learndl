import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class price_high_low_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'others'
    description = '最高最低价比值分位数'
    
    def calc_factor(self, date: int):
        """计算最高最低价比值的1年分位数
        1. 获取过去1年的最高最低价数据
        2. 计算比值的历史分位数
        """
        # TODO: 需要定义获取价格数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        high_prices = TRADE_DATA.get_high_price(start_date, date)
        low_prices = TRADE_DATA.get_low_price(start_date, date)
        
        ratio = high_prices / low_prices
        rank = ratio.rank(pct=True)
        
        return rank.rename('factor_value').to_frame() 