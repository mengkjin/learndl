import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class volpct_phigh(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '价格距离日内最高价的百分比'
    
    def calc_factor(self, date: int):
        """计算价格距离日内最高价的百分比
        1. 获取当日最高价和收盘价数据
        2. 计算百分比距离
        """
        # TODO: 需要定义获取价格数据的接口
        high_price = TRADE_DATA.get_high_price(date)
        close_price = TRADE_DATA.get_close_price(date)
        
        pct = (high_price - close_price) / high_price
        return pct.rename('factor_value').to_frame() 