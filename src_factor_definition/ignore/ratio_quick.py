import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ratio_quick(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'quality'
    description = '速动比率'
    
    def calc_factor(self, date: int):
        """计算速动比率
        1. 获取速动资产和流动负债数据
        2. 计算比值
        """
        # TODO: 需要定义获取资产负债表数据的接口
        current_assets = TRADE_DATA.get_balance_sheet(date, 'current_assets')
        inventory = TRADE_DATA.get_balance_sheet(date, 'inventory')
        current_liab = TRADE_DATA.get_balance_sheet(date, 'current_liabilities')
        
        quick_assets = current_assets - inventory
        ratio = quick_assets / current_liab
        return ratio.rename('factor_value').to_frame() 