import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class periodexp_ta_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '季度期间费用/总资产'
    
    def calc_factor(self, date: int):
        """计算季度期间费用资产比
        1. 获取季度期间费用和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        period_expense = TRADE_DATA.get_income_statement(date, 'period_expense', 'quarterly')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = period_expense / total_assets
        return ratio.rename('factor_value').to_frame() 