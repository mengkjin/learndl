import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tax_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax_equ环比变化值'
    
    def calc_factor(self, date: int):
        """计算所得税权益比的环比变化
        1. 获取当前和上季度的所得税和净资产数据
        2. 计算比率的环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        tax_curr = TRADE_DATA.get_income_statement(date, 'income_tax')
        equ_curr = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        tax_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'income_tax')
        equ_prev = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-1, 'Q'), 'total_equity')
        
        ratio_curr = tax_curr / equ_curr
        ratio_prev = tax_prev / equ_prev
        change = ratio_curr - ratio_prev
        
        return change.rename('factor_value').to_frame() 