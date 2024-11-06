import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_dedu_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '扣非净利润权益比'
    
    def calc_factor(self, date: int):
        """计算扣非净利润权益比
        1. 获取扣非净利润和所有者权益数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit_deducted = TRADE_DATA.get_income_statement(date, 'net_profit_deducted')
        total_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        ratio = net_profit_deducted / total_equity
        return ratio.rename('factor_value').to_frame() 