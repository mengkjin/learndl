import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class assetcur_asset(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'structure'
    description = '流动资产占比'
    
    def calc_factor(self, date: int):
        """计算流动资产占总资产比例
        1. 获取流动资产和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        current_assets = TRADE_DATA.get_balance_sheet(date, 'current_assets')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = current_assets / total_assets
        return ratio.rename('factor_value').to_frame() 