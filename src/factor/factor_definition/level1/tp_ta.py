import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tp_ta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '利润总额/总资产'
    
    def calc_factor(self, date: int):
        """计算利润总额资产比
        1. 获取利润总额和总资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_profit = TRADE_DATA.get_income_statement(date, 'total_profit')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        
        ratio = total_profit / total_assets
        return ratio.rename('factor_value').to_frame() 