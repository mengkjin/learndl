import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class liabcur_liab(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '流动负债占比'
    
    def calc_factor(self, date: int):
        """计算流动负债占总负债比例
        1. 获取流动负债和总负债数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        current_liabilities = TRADE_DATA.get_balance_sheet(date, 'current_liabilities')
        total_liabilities = TRADE_DATA.get_balance_sheet(date, 'total_liabilities')
        
        ratio = current_liabilities / total_liabilities
        return ratio.rename('factor_value').to_frame() 