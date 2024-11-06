import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class beta_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '日内beta'
    
    def calc_factor(self, date: int):
        """计算日内beta
        1. 获取日内分钟级别收益率数据
        2. 计算个股与指数的beta系数
        """
        # TODO: 需要定义获取日内数据的接口
        stock_ret = TRADE_DATA.get_intraday_returns(date)
        index_ret = TRADE_DATA.get_index_intraday_returns(date)
        
        # 计算beta
        beta = stock_ret.apply(lambda x: np.cov(x, index_ret)[0,1] / np.var(index_ret))
        return beta.rename('factor_value').to_frame() 