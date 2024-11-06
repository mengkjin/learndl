import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ep_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = '季度盈利收益率'
    
    def calc_factor(self, date: int):
        """计算季度盈利收益率
        1. 获取季度净利润和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit', 'quarterly')
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = net_profit / market_value
        return ratio.rename('factor_value').to_frame() 