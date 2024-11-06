import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class stop_q(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '季度营业收入收益率'
    
    def calc_factor(self, date: int):
        """计算季度营业收入收益率
        1. 获取季度营业收入和市值数据
        2. 计算比值
        """
        # TODO: 需要定义获取财务数据的接口
        sales = TRADE_DATA.get_income_statement(date, 'revenue', 'quarterly')
        market_value = TRADE_DATA.get_mkt_value(date)
        
        ratio = sales / market_value
        return ratio.rename('factor_value').to_frame() 