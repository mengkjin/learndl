import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_cratio(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '净利润覆盖率占比'
    
    def calc_factor(self, date: int):
        """计算净利润覆盖率
        1. 获取净利润和总收入数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        total_revenue = TRADE_DATA.get_income_statement(date, 'total_revenue')
        
        ratio = net_profit / total_revenue
        return ratio.rename('factor_value').to_frame() 