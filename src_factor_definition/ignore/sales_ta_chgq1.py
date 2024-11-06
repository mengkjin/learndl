import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sales_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sale_ta环比变化值'
    
    def calc_factor(self, date: int):
        """计算营业收入资产比的环比变化
        1. 获取当前和上季度的营业收入和总资产数据
        2. 计算比率的环比变化
        """
        # TODO: 需要定义获取财务数据的接口
        sales_curr = TRADE_DATA.get_income_statement(date, 'revenue')
        ta_curr = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        sales_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'revenue')
        ta_prev = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-1, 'Q'), 'total_assets')
        
        ratio_curr = sales_curr / ta_curr
        ratio_prev = sales_prev / ta_prev
        change = ratio_curr - ratio_prev
        
        return change.rename('factor_value').to_frame() 