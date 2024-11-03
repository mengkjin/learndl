import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class acc_eaa(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '应计利润率A'
    
    def calc_factor(self, date: int):
        """计算应计利润率A
        1. 获取净利润和经营现金流数据
        2. 计算应计利润率
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        op_cashflow = TRADE_DATA.get_cashflow(date, 'operating_cashflow')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        # 应计利润 = 净利润 - 经营现金流
        accrual = net_profit - op_cashflow
        ratio = accrual / total_assets
        
        return ratio.rename('factor_value').to_frame() 