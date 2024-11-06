import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sales_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '营业收入资产比'
    
    def calc_factor(self, date: int):
        """计算营业收入资产比
        1. 获取营业收入和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = revenue / total_assets
        return ratio.rename('factor_value').to_frame() 