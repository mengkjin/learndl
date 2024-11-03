import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class asset_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '总资产同比增长率'
    
    def calc_factor(self, date: int):
        """计算总资产同比增长率
        1. 获取当期和去年同期的总资产数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        prev_assets = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-4, 'Q'), 'total_assets')
        
        pct_change = (curr_assets - prev_assets) / abs(prev_assets)
        return pct_change.rename('factor_value').to_frame() 