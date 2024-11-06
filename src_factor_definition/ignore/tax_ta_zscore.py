import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class tax_ta_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'tax_ta标准化得分'
    
    def calc_factor(self, date: int):
        """计算所得税资产比的标准化得分
        1. 获取所得税和总资产数据
        2. 计算比值
        3. 按行业标准化
        """
        # TODO: 需要定义获取财务数据的接口
        tax = TRADE_DATA.get_income_statement(date, 'income_tax')
        total_assets = TRADE_DATA.get_balance_sheet(date, 'total_assets')
        industry = TRADE_DATA.get_industry(date)
        
        ratio = tax / total_assets
        
        # 按行业标准化
        def standardize(x): return (x - x.mean()) / x.std()
        zscore = ratio.groupby(industry).transform(standardize)
        
        return zscore.rename('factor_value').to_frame() 