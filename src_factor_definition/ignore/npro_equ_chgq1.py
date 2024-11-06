import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '净利润权益比环比变化'
    
    def calc_factor(self, date: int):
        """计算净利润权益比环比变化
        1. 获取当期和上期的净利润和所有者权益数据
        2. 计算环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        curr_np = TRADE_DATA.get_income_statement(date, 'net_profit')
        curr_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        prev_date = TradeDate(date).offset(-1, 'Q')
        prev_np = TRADE_DATA.get_income_statement(prev_date, 'net_profit')
        prev_equity = TRADE_DATA.get_balance_sheet(prev_date, 'total_equity')
        
        curr_ratio = curr_np / curr_equity
        prev_ratio = prev_np / prev_equity
        chg = (curr_ratio - prev_ratio) / abs(prev_ratio)
        
        return chg.rename('factor_value').to_frame() 