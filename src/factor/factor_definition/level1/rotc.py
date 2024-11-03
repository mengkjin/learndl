import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class rotc(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'EBIT/有形资产'
    
    def calc_factor(self, date: int):
        """计算有形资产回报率
        1. 获取EBIT和有形资产数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        ebit = TRADE_DATA.get_income_statement(date, 'ebit')
        tangible_assets = TRADE_DATA.get_balance_sheet(date, 'tangible_assets')
        
        ratio = ebit / tangible_assets
        return ratio.rename('factor_value').to_frame() 