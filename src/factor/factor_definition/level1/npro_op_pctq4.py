import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_op_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润同比变化率'
    
    def calc_factor(self, date: int):
        """计算营业利润同比增长率
        1. 获取当前和去年同期的营业利润数据
        2. 计算同比变化率
        """
        # TODO: 需要定义获取财务数据的接口
        op_curr = TRADE_DATA.get_income_statement(date, 'operating_profit')
        op_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'operating_profit')
        
        growth_rate = (op_curr - op_prev) / abs(op_prev)
        return growth_rate.rename('factor_value').to_frame() 