import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class skew_intra1min(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_volatility'
    description = '高频偏度'
    
    def calc_factor(self, date: int):
        """计算1分钟高频偏度
        1. 获取1分钟频率收益率数据
        2. 计算偏度
        """
        # TODO: 需要定义获取日内数据的接口
        ret_1min = TRADE_DATA.get_intraday_returns(date, freq='1min')
        skewness = ret_1min.apply(lambda x: pd.Series(x).skew())
        return skewness.rename('factor_value').to_frame() 