import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class assetcur_liabcur(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'liquidity'
    description = '流动资产/流动负债'
    
    def calc_factor(self, date: int):
        """计算流动比率
        1. 获取流动资产和流动负债数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        current_assets = TRADE_DATA.get_balance_sheet(date, 'current_assets')
        current_liab = TRADE_DATA.get_balance_sheet(date, 'current_liabilities')
        
        ratio = current_assets / current_liab
        return ratio.rename('factor_value').to_frame() 