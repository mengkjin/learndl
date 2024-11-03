import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ebitda_ev2(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'EBITDA/EV2'
    
    def calc_factor(self, date: int):
        """计算EBITDA/企业价值比率(另一种计算方式)
        1. 获取EBITDA和企业价值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        ebitda = TRADE_DATA.get_income_statement(date, 'ebitda')
        ev2 = TRADE_DATA.get_enterprise_value2(date)  # 另一种企业价值计算方式
        
        ratio = ebitda / ev2
        return ratio.rename('factor_value').to_frame() 