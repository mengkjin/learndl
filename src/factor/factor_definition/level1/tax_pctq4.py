import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tax_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税同比变化率'
    
    def calc_factor(self, date: int):
        """计算所得税同比增长率
        1. 获取当前和去年同期的所得税数据
        2. 计算同比变化率
        """
        # TODO: 需要定义获取财务数据的接口
        tax_curr = TRADE_DATA.get_income_statement(date, 'income_tax')
        tax_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'income_tax')
        
        growth_rate = (tax_curr - tax_prev) / abs(tax_prev)
        return growth_rate.rename('factor_value').to_frame() 