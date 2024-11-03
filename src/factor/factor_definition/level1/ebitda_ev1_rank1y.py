import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class ebitda_ev1_rank1y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'value'
    description = 'ebitda_ev1分位数'
    
    def calc_factor(self, date: int):
        """计算ebitda_ev1的1年分位数
        1. 获取过去1年的ebitda_ev1数据
        2. 计算当前值在历史分布中的分位数
        """
        # TODO: 需要定义获取财务数据的接口
        start_date = TradeDate(date).offset(-12, 'M')
        ebitda = TRADE_DATA.get_income_statement_series(start_date, date, 'ebitda')
        ev = TRADE_DATA.get_enterprise_value_series(start_date, date)
        
        ratio_series = ebitda / ev
        rank = ratio_series.groupby('secid').apply(lambda x: x.rank(pct=True).iloc[-1])
        
        return rank.rename('factor_value').to_frame() 