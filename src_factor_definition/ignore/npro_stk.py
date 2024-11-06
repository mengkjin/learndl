import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_stk(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '净利润/实收资本'
    
    def calc_factor(self, date: int):
        """计算净利润实收资本比
        1. 获取净利润和实收资本数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        paid_in_capital = TRADE_DATA.get_balance_sheet(date, 'paid_in_capital')
        
        ratio = net_profit / paid_in_capital
        return ratio.rename('factor_value').to_frame() 