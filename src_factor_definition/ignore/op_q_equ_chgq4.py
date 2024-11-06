import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class op_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'op_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        """计算季度营业利润权益比的同比变化
        1. 获取当前和去年同期的季度营业利润和净资产数据
        2. 计算比率的同比变化
        """
        # TODO: 需要定义获取财务数据的接口
        op_curr = TRADE_DATA.get_income_statement(date, 'operating_profit', 'quarterly')
        equ_curr = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        op_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'operating_profit', 'quarterly')
        equ_prev = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-4, 'Q'), 'total_equity')
        
        ratio_curr = op_curr / equ_curr
        ratio_prev = op_prev / equ_prev
        change = ratio_curr - ratio_prev
        
        return change.rename('factor_value').to_frame()