import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tax_sale(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '所得税/营业收入'
    
    def calc_factor(self, date: int):
        """计算所得税收入比
        1. 获取所得税和营业收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        tax = TRADE_DATA.get_income_statement(date, 'income_tax')
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        
        ratio = tax / revenue
        return ratio.rename('factor_value').to_frame() 