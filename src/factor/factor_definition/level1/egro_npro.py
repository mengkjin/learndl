import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class egro_npro(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净利润增长率'
    
    def calc_factor(self, date: int):
        """计算净利润增长率
        1. 获取当期和上期的净利润数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        prev_profit = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Y'), 'net_profit')
        
        growth_rate = (curr_profit - prev_profit) / abs(prev_profit)
        return growth_rate.rename('factor_value').to_frame()