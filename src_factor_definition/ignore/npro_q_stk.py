import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_q_stk(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '季报/实收资本'
    
    def calc_factor(self, date: int):
        """计算季度净利润与实收资本的比值
        1. 获取季度净利润和实收资本数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        npro = TRADE_DATA.get_income_statement(date, 'net_profit', 'quarterly')
        capital = TRADE_DATA.get_balance_sheet(date, 'paid_in_capital')
        
        ratio = npro / capital
        return ratio.rename('factor_value').to_frame() 