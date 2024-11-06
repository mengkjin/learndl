import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class roe(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '净资产收益率'
    
    def calc_factor(self, date: int):
        """计算净资产收益率
        1. 获取净利润和净资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        total_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        ratio = net_profit / total_equity
        return ratio.rename('factor_value').to_frame() 