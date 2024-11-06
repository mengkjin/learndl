import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class conf_persit(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内收益率持续性'
    
    def calc_factor(self, date: int):
        """计算日内收益率持续性
        1. 获取日内收益率数据
        2. 计算自相关系数
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        
        def calc_persistence(r):
            # 计算一阶自相关系数
            return pd.Series(r).autocorr(lag=1)
            
        persistence = returns.groupby('secid').apply(calc_persistence)
        return persistence.rename('factor_value').to_frame() 