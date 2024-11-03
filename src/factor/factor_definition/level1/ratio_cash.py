import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ratio_cash(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '现金比率'
    
    def calc_factor(self, date: int):
        """计算现金比率
        1. 获取货币资金和流动负债数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        cash = TRADE_DATA.get_balance_sheet(date, 'cash')
        current_liabilities = TRADE_DATA.get_balance_sheet(date, 'current_liabilities')
        
        ratio = cash / current_liabilities
        return ratio.rename('factor_value').to_frame() 