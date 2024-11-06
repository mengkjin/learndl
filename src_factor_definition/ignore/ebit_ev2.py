import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ebit_ev2(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBIT/EV2'
    
    def calc_factor(self, date: int):
        """计算EBIT/企业价值比率(另一种计算方式)
        1. 获取EBIT和企业价值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        ebit = TRADE_DATA.get_income_statement(date, 'ebit')
        ev2 = TRADE_DATA.get_enterprise_value2(date)  # 另一种企业价值计算方式
        
        ratio = ebit / ev2
        return ratio.rename('factor_value').to_frame() 