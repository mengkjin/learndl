import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class npro_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '业绩趋势增长率'
    
    def calc_factor(self, date: int):
        """计算净利润趋势增长率
        1. 获取最近4个季度的净利润数据
        2. 计算趋势增长率
        """
        # TODO: 需要定义获取财务数据的接口
        quarters = range(4)
        npro_data = pd.DataFrame()
        
        for i in quarters:
            offset_date = TradeDate(date).offset(-i, 'Q')
            npro = TRADE_DATA.get_income_statement(offset_date, 'net_profit', 'quarterly')
            npro_data[f'q{i}'] = npro
            
        # 计算趋势(使用简单线性回归斜率)
        trend = npro_data.apply(lambda x: np.polyfit(quarters, x, 1)[0], axis=1)
        
        return trend.rename('factor_value').to_frame() 