import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class liabnoncur_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '非流动负债/总资产'
    
    def calc_factor(self, date: int):
        """计算非流动负债资产比
        1. 获取非流动负债和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        noncurrent_liab = TRADE_DATA.get_balance_sheet(date, 'noncurrent_liabilities')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = noncurrent_liab / total_assets
        return ratio.rename('factor_value').to_frame() 