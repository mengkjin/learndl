import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class asset_size(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'size'
    description = '总资产规模'
    
    def calc_factor(self, date: int):
        """计算总资产规模
        1. 获取总资产数据
        2. 取对数
        """
        # TODO: 需要定义获取财务数据的接口
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        log_assets = np.log(total_assets)
        return log_assets.rename('factor_value').to_frame() 