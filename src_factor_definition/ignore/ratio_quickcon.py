import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ratio_quickcon(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '速动比率(含存货)'
    
    def calc_factor(self, date: int):
        """计算速动比率(含存货)
        1. 获取流动资产、存货和流动负债数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        current_assets = TRADE_DATA.get_balance_sheet(date, 'current_assets')
        inventory = TRADE_DATA.get_balance_sheet(date, 'inventory')
        current_liabilities = TRADE_DATA.get_balance_sheet(date, 'current_liabilities')
        
        ratio = (current_assets - inventory) / current_liabilities
        return ratio.rename('factor_value').to_frame() 