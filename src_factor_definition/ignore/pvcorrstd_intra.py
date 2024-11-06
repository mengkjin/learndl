import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class pvcorrstd_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'high_frequency'
    category1 = 'hf_correlation'
    description = '价量相关标准差'
    
    def calc_factor(self, date: int):
        """计算日内价量相关性的标准差
        1. 获取日内分钟级别价格和成交量数据
        2. 计算相关系数的标准差
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        # 计算价量相关性标准差
        def calc_pv_corr_std(p, v):
            corr = pd.rolling_corr(p, v, window=30)
            return np.std(corr)
            
        pv_corr_std = price.apply(lambda x: calc_pv_corr_std(x, volume))
        return pv_corr_std.rename('factor_value').to_frame() 