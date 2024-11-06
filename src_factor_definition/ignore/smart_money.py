import numpy as np
import pandas as pd
from src.factor.classes import StockFactorCalculator
from src.data.tushare.basic import CALENDAR, TRADE_DATA, TradeDate

class smart_money(StockFactorCalculator):
    init_date = 20070101
    category0 = 'technical'
    category1 = 'high_frequency'
    description = '聪明钱指标'
    
    def calc_factor(self, date: int):
        """计算聪明钱指标
        1. 获取日内不同时段的成交量和价格数据
        2. 计算指标值
        """
        # TODO: 需要定义获取日内数据的接口
        volume = TRADE_DATA.get_intraday_volume(date)
        price = TRADE_DATA.get_intraday_price(date)
        
        def calc_smart_money(v, p):
            # 定义聪明钱指标计算方法
            # 例如:开盘和收盘时段的成交量加权价格之差
            open_period = slice('09:30:00', '10:00:00')
            close_period = slice('14:30:00', '15:00:00')
            
            open_vwap = (v[open_period] * p[open_period]).sum() / v[open_period].sum()
            close_vwap = (v[close_period] * p[close_period]).sum() / v[close_period].sum()
            
            return close_vwap - open_vwap
            
        smart = volume.groupby('secid').apply(lambda x: calc_smart_money(x, price))
        return smart.rename('factor_value').to_frame()