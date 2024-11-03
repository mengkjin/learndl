import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tp_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tp_equ环比变化值'
    
    def calc_factor(self, date: int):
        """计算利润总额权益比的环比变化
        1. 获取当前和上季度的利润总额和净资产数据
        2. 计算比率的环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        tp_curr = TRADE_DATA.get_income_statement(date, 'total_profit')
        equ_curr = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        tp_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'total_profit')
        equ_prev = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-1, 'Q'), 'total_equity')
        
        ratio_curr = tp_curr / equ_curr
        ratio_prev = tp_prev / equ_prev
        change = ratio_curr - ratio_prev
        
        return change.rename('factor_value').to_frame() 