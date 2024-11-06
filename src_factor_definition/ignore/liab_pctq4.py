import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class liab_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '负债同比增长率'
    
    def calc_factor(self, date: int):
        """计算负债同比增长率
        1. 获取当期和去年同期的负债数据
        2. 计算增长率
        """
        # TODO: 需要定义获取财务数据的接口
        curr_liab = TRADE_DATA.get_balance_sheet(date, 'total_liabilities')
        prev_liab = TRADE_DATA.get_balance_sheet(TradeDate(date).offset(-4, 'Q'), 'total_liabilities')
        
        pct_change = (curr_liab - prev_liab) / abs(prev_liab)
        return pct_change.rename('factor_value').to_frame() 