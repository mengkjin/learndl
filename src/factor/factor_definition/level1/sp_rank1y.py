import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sp_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'sp分位数'
    
    def calc_factor(self, date: int):
        """计算营业收入市值比的1年分位数
        1. 获取过去1年的营业收入和市值数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        revenue = TRADE_DATA.get_income_statement(date, 'revenue')
        market_value = TRADE_DATA.get_market_value(date)
        
        ratio = revenue / market_value
        return ratio.rename('factor_value').to_frame() 