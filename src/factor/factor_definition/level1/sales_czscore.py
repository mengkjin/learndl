import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class sales_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sales增速与同行业差的标准化得分'
    
    def calc_factor(self, date: int):
        """计算营业收入增速的行业标准化得分
        1. 获取营业收入数据
        2. 计算同比增速
        3. 按行业计算标准化得分
        """
        # TODO: 需要定义获取财务数据的接口
        sales_curr = TRADE_DATA.get_income_statement(date, 'revenue')
        sales_prev = TRADE_DATA.get_income_statement(TradeDate(date).offset(-4, 'Q'), 'revenue')
        industry = TRADE_DATA.get_industry(date)
        
        # 计算同比增速
        growth = (sales_curr - sales_prev) / abs(sales_prev)
        
        # 按行业标准化
        def standardize(x): return (x - x.mean()) / x.std()
        zscore = growth.groupby(industry).transform(standardize)
        
        return zscore.rename('factor_value').to_frame() 