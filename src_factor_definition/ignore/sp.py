import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '营业收入/市值'
    
    def calc_factor(self, date: int):
        """计算营业收入市值比
        1. 获取营业收入和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = revenue / market_value
        return ratio.rename('factor_value').to_frame() 