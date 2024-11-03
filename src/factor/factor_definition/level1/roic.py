import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class roic(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'EBIT(1-税率)/投资资本'
    
    def calc_factor(self, date: int):
        """计算投资资本回报率
        1. 获取EBIT、税率和投资资本数据
        2. 计算ROIC
        """
        # TODO: 需要定义获取财务数据的接口
        ebit = TRADE_DATA.get_income_statement(date, 'ebit')
        tax_rate = TRADE_DATA.get_income_statement(date, 'tax_rate')
        invested_capital = TRADE_DATA.get_balance_sheet(date, 'invested_capital')
        
        roic = ebit * (1 - tax_rate) / invested_capital
        return roic.rename('factor_value').to_frame() 