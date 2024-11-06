import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class corr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内收益率与成交量相关系数'
    
    def calc_factor(self, date: int):
        """计算日内收益率与成交量相关系数
        1. 获取日内收益率和成交量数据
        2. 计算相关系数
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_correlation(r, v):
            # 计算相关系数
            return pd.Series(r).corr(pd.Series(v))
            
        corr = returns.groupby('secid').apply(lambda x: calc_correlation(x, volume[x.index]))
        return corr.rename('factor_value').to_frame() 