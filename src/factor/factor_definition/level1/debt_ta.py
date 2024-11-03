import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class debt_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '有息负债/总资产'
    
    def calc_factor(self, date: int):
        """计算有息负债资产比
        1. 获取有息负债和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        interest_bearing_debt = TRADE_DATA.get_balance_sheet(date, 'interest_bearing_debt')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = interest_bearing_debt / total_assets
        return ratio.rename('factor_value').to_frame() 