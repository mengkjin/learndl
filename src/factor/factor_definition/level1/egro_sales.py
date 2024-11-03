import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class egro_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入增长率'
    
    def calc_factor(self, date: int):
        """计算营业收入增长率
        1. 获取当期和上期的营业收入数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_sales = TRADE_DATA.get_income_statement(date, 'revenue')
        prev_sales = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Y'), 'revenue')
        
        growth_rate = (curr_sales - prev_sales) / abs(prev_sales)
        return growth_rate.rename('factor_value').to_frame() 