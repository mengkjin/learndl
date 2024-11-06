import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_highret(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '高收益率成交量'
    
    def calc_factor(self, date: int):
        """计算高收益率成交量
        1. 获取日内收益率和成交量数据
        2. 计算高收益率时段的成交量占比
        """
        # TODO: 需要定义获取日内数据的接口
        returns = TRADE_DATA.get_intraday_returns(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_high_ret_vol(r, v):
            # 定义高收益率阈值(75分位数)
            high_ret_threshold = r.quantile(0.75)
            # 计算高收益率时段的成交量占比
            high_ret_vol = v[r >= high_ret_threshold].sum() / v.sum()
            return high_ret_vol
            
        ratio = volume.groupby('secid').apply(lambda x: calc_high_ret_vol(returns[x.index], x))
        return ratio.rename('factor_value').to_frame() 