import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_highvwap(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '高VWAP成交量'
    
    def calc_factor(self, date: int):
        """计算高VWAP成交量
        1. 获取日内VWAP和成交量数据
        2. 计算高VWAP时段的成交量占比
        """
        # TODO: 需要定义获取日内数据的接口
        vwap = TRADE_DATA.get_intraday_vwap(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_high_vwap_vol(p, v):
            # 定义高VWAP阈值(75分位数)
            high_vwap_threshold = p.quantile(0.75)
            # 计算高VWAP时段的成交量占比
            high_vwap_vol = v[p >= high_vwap_threshold].sum() / v.sum()
            return high_vwap_vol
            
        ratio = volume.groupby('secid').apply(lambda x: calc_high_vwap_vol(vwap[x.index], x))
        return ratio.rename('factor_value').to_frame()