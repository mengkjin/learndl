import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class liab_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '总资产负债率'
    
    def calc_factor(self, date: int):
        """计算总资产负债率
        1. 获取负债总额和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_liabilities = TRADE_DATA.get_balance_sheet(date, 'total_liabilities')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = total_liabilities / total_assets
        return ratio.rename('factor_value').to_frame() 