import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class periodexp_q_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '期间费用/营业收入'
    
    def calc_factor(self, date: int):
        """计算季度期间费用收入比
        1. 获取季度期间费用和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        period_expense = TRADE_DATA.get_income_statement(date, 'period_expense', 'quarterly')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue', 'quarterly')
        
        ratio = period_expense / revenue
        return ratio.rename('factor_value').to_frame() 