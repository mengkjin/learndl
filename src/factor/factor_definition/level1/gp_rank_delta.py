import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class gp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润排名变化'
    
    def calc_factor(self, date: int):
        """计算毛利润排名变化
        1. 获取当期和上期的毛利润数据
        2. 计算排名变化
        """
        # TODO: 需要定义获取财务数据的接口
        curr_gp = TRADE_DATA.get_income_statement(date, 'gross_profit')
        prev_gp = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'gross_profit')
        
        curr_rank = curr_gp.rank(pct=True)
        prev_rank = prev_gp.rank(pct=True)
        delta = curr_rank - prev_rank
        
        return delta.rename('factor_value').to_frame() 