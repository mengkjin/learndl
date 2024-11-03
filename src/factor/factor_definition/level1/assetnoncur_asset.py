import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class assetnoncur_asset(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'structure'
    description = '非流动资产占比'
    
    def calc_factor(self, date: int):
        """计算非流动资产占总资产比例
        1. 获取非流动资产和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        noncurrent_assets = TRADE_DATA.get_balance_sheet(date, 'noncurrent_assets')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = noncurrent_assets / total_assets
        return ratio.rename('factor_value').to_frame() 