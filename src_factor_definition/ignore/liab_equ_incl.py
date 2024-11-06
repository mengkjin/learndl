import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class liab_equ_incl(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'leverage'
    description = '资产负债率(含少数股东权益)'
    
    def calc_factor(self, date: int):
        """计算资产负债率(含少数股东权益)
        1. 获取负债总额和所有者权益(含少数股东权益)数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        total_liabilities = TRADE_DATA.get_balance_sheet(date, 'total_liabilities')
        total_equity_incl = TRADE_DATA.get_balance_sheet(date, 'total_equity_incl')
        
        ratio = total_liabilities / total_equity_incl
        return ratio.rename('factor_value').to_frame()