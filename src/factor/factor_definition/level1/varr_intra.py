import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class varr_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内方差比率'
    
    def calc_factor(self, date: int):
        """计算日内方差比率
        1. 获取日内价格数据
        2. 计算不同频率的方差比
        """
        # TODO: 需要定义获取日内数据的接口
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_variance_ratio(p):
            # 计算不同频率的收益率方差比
            ret_1min = np.log(p / p.shift(1))
            ret_5min = np.log(p / p.shift(5))
            
            var_1min = ret_1min.var()
            var_5min = ret_5min.var() / 5  # 标准化
            
            return var_5min / var_1min if var_1min != 0 else np.nan
            
        vr = price.groupby('secid').apply(calc_variance_ratio)
        return vr.rename('factor_value').to_frame() 