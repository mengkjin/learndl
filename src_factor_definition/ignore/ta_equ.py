import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ta_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'structure'
    description = '总资产权益比'
    
    def calc_factor(self, date: int):
        """计算总资产权益比
        1. 获取总资产和所有者权益数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        total_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        
        ratio = total_assets / total_equity
        return ratio.rename('factor_value').to_frame() 