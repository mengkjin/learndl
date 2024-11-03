import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class lpnp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '上期净利润'
    
    def calc_factor(self, date: int):
        """获取上期净利润
        1. 获取上期净利润数据
        """
        # TODO: 需要定义获取财务数据的接口
        prev_date = TradeDate(date).offset(-1, 'Q')
        prev_net_profit = TRADE_DATA.get_income_statement(prev_date, 'net_profit')
        
        return prev_net_profit.rename('factor_value').to_frame()