import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_tp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额在行业内分位数之差'
    
    def calc_factor(self, date: int):
        """计算利润总额行业分位数的变化
        1. 获取当前和上期的利润总额数据
        2. 计算行业内分位数的变化
        """
        # TODO: 需要定义获取财务数据的接口
        tp_curr = TRADE_DATA.get_income_statement(date, 'total_profit')
        tp_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-1, 'Q'), 'total_profit')
        industry = TRADE_DATA.get_industry(date)
        
        def calc_industry_rank(x): return x.rank(pct=True)
        rank_curr = tp_curr.groupby(industry).transform(calc_industry_rank)
        rank_prev = tp_prev.groupby(industry).transform(calc_industry_rank)
        
        rank_delta = rank_curr - rank_prev
        return rank_delta.rename('factor_value').to_frame() 