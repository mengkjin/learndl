import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tp_q_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '利润总额/净资产'
    
    def calc_factor(self, date: int):
        """计算季度利润总额权益比
        1. 获取季度利润总额和净资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_profit = TRADE_DATA.get_income_statement(date, 'total_profit', 'quarterly')
        total_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        ratio = total_profit / total_equity
        return ratio.rename('factor_value').to_frame() 