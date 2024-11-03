import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vol_lowvwap(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '低VWAP成交量'
    
    def calc_factor(self, date: int):
        """计算低VWAP成交量
        1. 获取日内VWAP和成交量数据
        2. 计算低VWAP时段的成交量占比
        """
        # TODO: 需要定义获取日内数据的接口
        vwap = TRADE_DATA.get_intraday_vwap(date)
        volume = TRADE_DATA.get_intraday_volume(date)
        
        def calc_low_vwap_vol(p, v):
            # 定义低VWAP阈值(25分位数)
            low_vwap_threshold = p.quantile(0.25)
            # 计算低VWAP时段的成交量占比
            low_vwap_vol = v[p <= low_vwap_threshold].sum() / v.sum()
            return low_vwap_vol
            
        ratio = volume.groupby('secid').apply(lambda x: calc_low_vwap_vol(vwap[x.index], x))
        return ratio.rename('factor_value').to_frame()