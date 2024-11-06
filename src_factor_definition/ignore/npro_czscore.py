import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'profitability'
    description = '净利润Z分数'
    
    def calc_factor(self, date: int):
        """计算净利润的行业标准化得分
        1. 获取净利润数据
        2. 按行业标准化
        """
        # TODO: 需要定义获取财务数据的接口
        net_profit = TRADE_DATA.get_income_statement(date, 'net_profit')
        industry = TRADE_DATA.get_industry(date)
        
        # 按行业标准化
        def standardize(x): return (x - x.mean()) / x.std()
        zscore = net_profit.groupby(industry).transform(standardize)
        
        return zscore.rename('factor_value').to_frame()