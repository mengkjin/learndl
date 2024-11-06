import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class egro_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净资产增长率'
    
    def calc_factor(self, date: int):
        """计算净资产增长率
        1. 获取当期和上期的净资产数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_equity = TRADE_DATA.get_balance_sheet(date, 'total_equity')
        prev_equity = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-1, 'Y'), 'total_equity')
        
        growth_rate = (curr_equity - prev_equity) / abs(prev_equity)
        return growth_rate.rename('factor_value').to_frame() 