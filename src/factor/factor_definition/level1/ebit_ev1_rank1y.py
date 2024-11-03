import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ebit_ev1_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'ebit_ev1分位数'
    
    def calc_factor(self, date: int):
        """计算ebit_ev1的1年分位数
        1. 获取过去1年的ebit_ev1数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date) - 240  # 约1年的交易日
        ebit = TRADE_DATA.get_income_statement(date, 'ebit')
        ev = TRADE_DATA.get_enterprise_value(date)
        
        ratio = ebit / ev
        rank = ratio.rank(pct=True)
        return rank.rename('factor_value').to_frame()