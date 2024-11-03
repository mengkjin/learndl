import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '毛利润资产比'
    
    def calc_factor(self, date: int):
        """计算毛利润资产比
        1. 获取毛利润和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        gross_profit = TRADE_DATA.get_income_statement(date, 'gross_profit')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = gross_profit / total_assets
        return ratio.rename('factor_value').to_frame() 