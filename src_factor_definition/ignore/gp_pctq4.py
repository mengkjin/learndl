import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润同比增长率'
    
    def calc_factor(self, date: int):
        """计算毛利润同比增长率
        1. 获取当期和去年同期的毛利润数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_gp = TRADE_DATA.get_income_statement(date, 'gross_profit')
        prev_gp = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'gross_profit')
        
        pct_change = (curr_gp - prev_gp) / abs(prev_gp)
        return pct_change.rename('factor_value').to_frame()