import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class cash_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'liquidity'
    description = '现金及现金等价物/总资产'
    
    def calc_factor(self, date: int):
        """计算现金资产比
        1. 获取现金及现金等价物和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        cash = TRADE_DATA.get_balance_sheet(date, 'cash_equivalents')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = cash / total_assets
        return ratio.rename('factor_value').to_frame() 