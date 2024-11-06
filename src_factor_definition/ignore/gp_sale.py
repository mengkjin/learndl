import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '毛利率'
    
    def calc_factor(self, date: int):
        """计算毛利率
        1. 获取毛利润和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        gross_profit = TRADE_DATA.get_income_statement(date, 'gross_profit')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        
        ratio = gross_profit / revenue
        return ratio.rename('factor_value').to_frame() 