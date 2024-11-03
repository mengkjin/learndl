import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class bm(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '账面市值比'
    
    def calc_factor(self, date: int):
        """计算账面市值比
        1. 获取净资产和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        book_value = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = book_value / market_value
        return ratio.rename('factor_value').to_frame() 