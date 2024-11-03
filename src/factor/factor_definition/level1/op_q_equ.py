import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class op_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '营业利润/净资产'
    
    def calc_factor(self, date: int):
        """计算季度营业利润权益比
        1. 获取季度营业利润和净资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        operating_profit = TRADE_DATA.get_income_statement(date, 'operating_profit', 'quarterly')
        total_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        ratio = operating_profit / total_equity
        return ratio.rename('factor_value').to_frame() 