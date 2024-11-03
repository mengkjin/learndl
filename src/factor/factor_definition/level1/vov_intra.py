import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class vov_intra(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '日内成交量波动率的波动率'
    
    def calc_factor(self, date: int):
        """计算日内成交量波动率的波动率
        1. 获取日内成交量数据
        2. 计算波动率的波动率
        """
        # TODO: 需要定义获取日内数据的接口
        start_date = TradeDate(date) - 20  # 约1个月的交易日
        volume = TRADE_DATA.get_intraday_volume(start_date, date)
        
        def calc_vov(v):
            # 计算每日的成交量波动率
            daily_vol = v.groupby(level=0).std()
            # 计算波动率的波动率
            return daily_vol.std()
            
        vov = volume.groupby('secid').apply(calc_vov)
        return vov.rename('factor_value').to_frame() 